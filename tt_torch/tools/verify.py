# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import onnx
import os
import tt_mlir
from tt_torch.onnx_compile import compile_onnx
from tt_torch.tools.utils import (
    onnx_output_to_torch,
    prepare_inference_session,
    run_model_proto,
)
from tt_torch.dynamo.backend import BackendOptions
from tt_torch.tools.utils import (
    calculate_atol,
    calculate_pcc,
    with_torch_dynamo_cleanup,
    CompileDepth,
    CompilerConfig,
)
from tt_torch.tools.device_manager import DeviceManager


def compile_model(model, compiler_config, device, async_mode):
    torch_options = BackendOptions()
    torch_options.compiler_config = compiler_config
    torch_options.devices = [device]
    torch_options.async_mode = async_mode

    if int(
        os.environ.get("TT_TORCH_FORCE_EXPERIMENTAL_BACKEND", False)
    ):
        backend = "tt-experimental"
    else:
        backend = "tt"

    return torch.compile(model, backend=backend, options=torch_options)


def generate_inputs(input_shapes, input_data_types, input_range, input_range_int):
    if all([dtype.is_floating_point for dtype in input_data_types]):
        low, high = input_range
        # Uniformly distribute random numbers within the input_range
        inputs = [
            (low - high) * torch.rand(shape, dtype=dtype) + high
            for shape, dtype in zip(input_shapes, input_data_types)
        ]
    elif all([dtype == torch.bool for dtype in input_data_types]):
        inputs = [
            torch.randint(0, 2, shape, dtype=torch.bool) for shape in input_shapes
        ]
    else:
        low, high = input_range_int
        inputs = [
            torch.randint(low, high, shape, dtype=torch.int32) for shape in input_shapes
        ]
    return inputs


def verify_against_golden(
    golden_tensors,
    calculated_tensors,
    assert_pcc,
    assert_atol,
    required_pcc=0.99,
    required_atol=None,
    relative_atol=None,
    disable_print=False,
):
    assert (required_atol is not None) != (
        relative_atol is not None
    ), "Exactly one of atol or relative_atol should be provided."
    assert isinstance(
        golden_tensors, tuple
    ), f"Expecting the golden tensors to be a tuple of tensors after _extract_outputs. Got type: {type(golden_tensors)}"
    assert isinstance(
        calculated_tensors, tuple
    ), f"Expecting the calculated tensors to be a tuple of tensors after _extract_outputs. Got type: {type(calculated_tensors)}"
    assert len(golden_tensors) == len(
        calculated_tensors
    ), f"Expecting the number of golden tensors (ct: {len(golden_tensors)}) and calculated tensors (ct: {len(calculated_tensors)}) to be the same."

    pccs, pcc_passeds = [], []
    atols, atol_thresholds, atols_passeds = [], [], []

    # Distinct value to put in the `pccs` list so we can append the correct log
    SKIPPED_PCC_CALCULATION_FOR_SINGLE_VALUE = None
    SKIPPED_NON_TENSOR_ITEM = None

    for i, (golden, calculated) in enumerate(zip(golden_tensors, calculated_tensors)):
        if not isinstance(golden, torch.Tensor) or not isinstance(
            calculated, torch.Tensor
        ):
            # For non-tensor items, check exact equality
            if golden == calculated:
                pccs.append(SKIPPED_NON_TENSOR_ITEM)
                pcc_passeds.append(True)
                atols.append(SKIPPED_NON_TENSOR_ITEM)
                atol_thresholds.append(0)
                atols_passeds.append(True)
            else:
                # Items don't match
                pccs.append(SKIPPED_NON_TENSOR_ITEM)
                pcc_passeds.append(False)
                atols.append(SKIPPED_NON_TENSOR_ITEM)
                atol_thresholds.append(0)
                atols_passeds.append(False)
            continue
        assert (
            golden.shape == calculated.shape
        ), f"Shape mismatch on output {i}: {golden.shape} vs {calculated.shape}"
        assert isinstance(golden, torch.Tensor) and isinstance(
            calculated, torch.Tensor
        ), f"Expecting both golden and calculated tensors to be of type torch.Tensor for output {i}, but got golden: {type(golden)} and calculated: {type(calculated)}"

        if golden.flatten().size() == (1,):
            pcc_ = SKIPPED_PCC_CALCULATION_FOR_SINGLE_VALUE
            pcc_passeds.append(True)
        else:
            pcc_ = calculate_pcc(golden, calculated)
            pcc_passeds.append(pcc_ >= required_pcc)
        pccs.append(pcc_)

        if relative_atol is not None:
            max_value = (torch.max(torch.abs(golden[~torch.isnan(golden)]))).item()
            required_atol = max_value * relative_atol

        atol_thresholds.append(required_atol)
        atol_ = calculate_atol(golden, calculated)
        atols.append(atol_)
        atols_passeds.append(atol_ <= required_atol)

    check_mark = "\U00002705"
    red_x = "\U0000274C"
    warning = "\U0000274E"

    pcc_warning = f"{warning} (assert_pcc == False)"
    atol_warning = f"{warning} (assert_atol == False)"

    passed_pcc = True
    passed_atol = True
    err_msg = ""
    msg = ""

    # Iterate over all tensors and their results and build a message for printing.
    for i, ((pcc_passed, pcc_), (atol_passed, atol_), atol_threshold) in enumerate(
        zip(zip(pcc_passeds, pccs), zip(atols_passeds, atols), atol_thresholds)
    ):
        msg = msg + f"Results for output {i}:\n"
        if pcc_passed:
            if (
                pcc_ != SKIPPED_PCC_CALCULATION_FOR_SINGLE_VALUE
                or pcc_ != SKIPPED_NON_TENSOR_ITEM
            ):
                msg = msg + f"  PCC: {pcc_:0,.4f}, threshold: {required_pcc} "
                msg = msg + f"{check_mark}\n"
        else:
            msg = msg + f"  PCC: {pcc_:0,.4f}, threshold: {required_pcc} "
            msg = msg + f"{red_x if assert_pcc else pcc_warning}\n"
            err_msg = (
                err_msg
                + f"PCC of output {i}: {pcc_:0,.4f}, threshold: {required_pcc} {red_x if assert_pcc else pcc_warning}\n"
            )
            passed_pcc = False

        if atol_passed:
            if atol_ != SKIPPED_NON_TENSOR_ITEM:
                msg = (
                    msg
                    + f"  ATOL: {atol_:0,.4f}, threshold: {atol_threshold}{f' (calculated using relative_atol: {relative_atol})' if relative_atol is not None else ''} "
                )
                msg = msg + f"{check_mark}\n"
            msg = msg + f"{check_mark}\n"
        else:
            msg = msg + f"{red_x if assert_atol else atol_warning}\n"

            err_msg = (
                err_msg
                + f"ATOL of output {i}: {atol_:0,.4f}, threshold: {atol_threshold}{f' (calculated using relative_atol: {relative_atol})' if relative_atol is not None else ''} {red_x if assert_atol else atol_warning}\n"
            )
            passed_atol = False

    # Now that all tensors are checked, print the final message
    if assert_pcc and assert_atol:
        if not passed_pcc or not passed_atol:
            assert False, err_msg
    elif not assert_pcc and assert_atol:
        print("Ignoring PCC check\n")
        if not passed_atol:
            assert False, err_msg
    elif assert_pcc and not assert_atol:
        print("Ignoring ATOL check\n")
        if not passed_pcc:
            assert False, err_msg
    if not disable_print:
        print(msg)
    return pccs, atols, passed_pcc, passed_atol


def _verify_torch_module(
    mod,
    inputs,
    input_shapes,
    input_data_types,
    required_pcc,
    required_atol,
    input_range,
    input_range_int,
    compiler_config,
    do_assert,
    device,
):
    if input_data_types is None:
        input_data_types = [torch.float32] * (
            len(input_shapes) if input_shapes is not None else len(inputs)
        )

    from tt_torch.dynamo.backend import backend  # avoid circular import

    if inputs is None:
        inputs = generate_inputs(
            input_shapes, input_data_types, input_range, input_range_int
        )

    golden = mod(*inputs)
    tt_mod = compile_model(mod, compiler_config, device, async_mode=False)

    ret = tt_mod(*inputs)

    if isinstance(golden, torch.Tensor):
        golden = (golden,)
    if isinstance(ret, torch.Tensor):
        ret = (ret,)

    # Incase they are lists
    golden = tuple(golden)
    ret = tuple(ret)

    verify_against_golden(
        golden, ret, do_assert, do_assert, required_pcc, required_atol=required_atol
    )


def _verify_onnx_module(
    model_proto: onnx.ModelProto,
    inputs,
    input_data_types,
    required_pcc,
    required_atol,
    input_range,
    input_range_int,
    compiler_config,
    do_assert,
):
    sess = prepare_inference_session(model_proto=model_proto)
    input_shapes = [nodearg.shape for nodearg in sess.get_inputs()]
    if input_data_types is None:
        input_data_types = [torch.float32] * (
            len(input_shapes) if input_shapes is not None else len(inputs)
        )

    if inputs is None:
        if all([dtype.is_floating_point for dtype in input_data_types]):
            low, high = input_range
            # Uniformly distribute random numbers within the input_range
            inputs = [
                (low - high) * torch.rand(shape, dtype=dtype) + high
                for shape, dtype in zip(input_shapes, input_data_types)
            ]
        else:
            low, high = input_range_int
            inputs = [
                torch.randint(low, high, shape, dtype=torch.int64)
                for shape in input_shapes
            ]
    golden = run_model_proto(
        model_proto=model_proto,
        sess=sess,
        inputs=inputs,
        input_data_types=input_data_types,
    )
    golden = onnx_output_to_torch(golden)
    compiled_mod = compile_onnx(model_proto, compiler_config)
    ret = compiled_mod(*inputs)
    if compiler_config.compile_depth not in [
        CompileDepth.EXECUTE,
        CompileDepth.EXECUTE_OP_BY_OP,
    ]:
        ret = onnx_output_to_torch(ret)

    assert len(golden) == len(
        ret
    ), f"Number of outputs mismatch between golden and compiled: {len(golden)} vs {len(ret)}"

    if isinstance(golden, torch.Tensor):
        golden = (golden,)
    if isinstance(ret, torch.Tensor):
        ret = (ret,)

    # Incase they are lists
    golden = tuple(golden)
    ret = tuple(ret)

    verify_against_golden(
        golden, ret, do_assert, do_assert, required_pcc, required_atol=required_atol
    )


@with_torch_dynamo_cleanup
def verify_module(
    mod,
    inputs=None,
    input_shapes=None,
    input_data_types=None,
    required_pcc=0.99,
    required_atol=1e-2,
    input_range=(-0.5, 0.5),
    input_range_int=(0, 1000),
    compiler_config=CompilerConfig(),
    do_assert=True,
    device=None,
):

    if isinstance(mod, torch.nn.Module):
        assert (
            input_shapes is not None or inputs is not None
        ), "Either input_shapes or inputs must be provided"
        _verify_torch_module(
            mod,
            inputs,
            input_shapes,
            input_data_types,
            required_pcc,
            required_atol,
            input_range,
            input_range_int,
            compiler_config,
            do_assert,
            device,
        )
    elif isinstance(mod, onnx.ModelProto):
        assert (
            input_shapes is None
        ), "When verifying an ONNX module, input_shapes must be None as they are inferred from the ONNX model"
        _verify_onnx_module(
            mod,
            inputs,
            input_data_types,
            required_pcc,
            required_atol,
            input_range,
            input_range_int,
            compiler_config,
            do_assert,
        )
    else:
        raise ValueError("Invalid module type")


@with_torch_dynamo_cleanup
def verify_torch_module_async(
    mod,
    inputs=None,
    input_shapes=None,
    input_data_types=None,
    required_pcc=0.99,
    required_atol=1e-2,
    input_range=(-0.5, 0.5),
    input_range_int=(0, 1000),
    compiler_config=CompilerConfig(),
    do_assert=True,
):
    assert (
        input_shapes is not None or inputs is not None
    ), "Either input_shapes or inputs must be provided"
    parent, devices = DeviceManager.acquire_available_devices()
    if input_data_types is None:
        input_data_types = [torch.float32] * (
            len(input_shapes) if input_shapes is not None else len(inputs)
        )
    tt_mods = []
    for device in devices:
        tt_mods.append(compile_model(mod, compiler_config, device, async_mode=True))
    if inputs is None:
        inputs = generate_inputs(
            input_shapes, input_data_types, input_range, input_range_int
        )
    golden = mod(*inputs)
    if isinstance(golden, torch.Tensor):
        golden = (golden,)
    golden = tuple(golden)
    rt_tensors = []
    for tt_mod in tt_mods:
        rt_tensors.append(tt_mod(*inputs))
    rets = []
    for rt_tensor in rt_tensors:
        ret = tt_mlir.to_host(rt_tensor)
        if isinstance(ret, torch.Tensor):
            ret = (ret,)
        ret = tuple(ret)
        rets.append(ret)

    for ret in rets:
        try:
            verify_against_golden(
                golden,
                ret,
                do_assert,
                do_assert,
                required_pcc,
                required_atol=required_atol,
            )
        except:
            break
    DeviceManager.release_parent_device(parent, cleanup_sub_devices=True)
