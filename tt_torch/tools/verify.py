# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import onnx
from onnxruntime import InferenceSession
import numpy as np
import tt_mlir
from tt_torch.onnx_compile import compile_onnx
from tt_torch.tools.utils import calculate_atol, calculate_pcc


def verify_against_golden(
    golden_tensors,
    calculated_tensors,
    required_pcc=0.99,
    required_atol=None,
    relative_atol=None,
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
    ), "Expecting the number of golden and calculated tensors to be the same."

    pccs, pcc_passeds = [], []
    atols, atol_thresholds, atols_passeds = [], [], []

    # Distinct value to put in the `pccs` list so we can append the correct log
    SKIPPED_PCC_CALCULATION_FOR_SINGLE_VALUE = None

    for i, (golden, calculated) in enumerate(zip(golden_tensors, calculated_tensors)):
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

    passed_pcc = True
    passed_atol = True
    err_msg = ""
    msg = ""
    for i, ((pcc_passed, pcc_), (atol_passed, atol_), atol_threshold) in enumerate(
        zip(zip(pcc_passeds, pccs), zip(atols_passeds, atols), atol_thresholds)
    ):
        msg = msg + f"Results for output {i}:\n"
        if pcc_passed:
            if pcc_ != SKIPPED_PCC_CALCULATION_FOR_SINGLE_VALUE:
                msg = (
                    msg
                    + f"  PCC: {pcc_:0,.4f}, threshold: {required_pcc} {check_mark}\n"
                )
        else:
            msg = msg + f"  PCC: {pcc_:0,.4f}, threshold: {required_pcc} {red_x}\n"
            err_msg = (
                err_msg
                + f"PCC of output {i}: {pcc_:0,.4f}, threshold: {required_pcc} {red_x}\n"
            )
            passed_pcc = False

        if atol_passed:
            msg = (
                msg
                + f"  ATOL: {atol_:0,.4f}, threshold: {atol_threshold}{f' (calculated using relative_atol: {relative_atol})' if relative_atol is not None else ''} {check_mark}\n"
            )
        else:
            msg = (
                msg
                + f"  ATOL: {atol_:0,.4f}, threshold: {atol_threshold}{f' (calculated using relative_atol: {relative_atol})' if relative_atol is not None else ''} {red_x}\n"
            )
            err_msg = (
                err_msg
                + f"ATOL of output {i}: {atol_:0,.4f}, threshold: {atol_threshold}{f' (calculated using relative_atol: {relative_atol})' if relative_atol is not None else ''} {red_x}\n"
            )
            passed_atol = False

    return passed_pcc, passed_atol, msg, err_msg, pccs, atols


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
):
    from tt_torch.dynamo.backend import backend  # avoid circular import

    if input_data_types is None:
        input_data_types = [torch.float32] * (
            len(input_shapes) if input_shapes is not None else len(inputs)
        )

    tt_mod = torch.compile(mod, backend=backend, options=compiler_config)
    if inputs is None:
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
                torch.randint(low, high, shape, dtype=torch.int32)
                for shape in input_shapes
            ]

    ret = tt_mod(*inputs)
    golden = mod(*inputs)

    if isinstance(golden, torch.Tensor):
        golden = (golden,)
    if isinstance(ret, torch.Tensor):
        ret = (ret,)

    # Incase they are lists
    golden = tuple(golden)
    ret = tuple(ret)

    passed_pcc, passed_atol, msg, err_msg, _, _ = verify_against_golden(
        golden, ret, required_pcc, required_atol=required_atol
    )
    print(msg)
    if do_assert:
        assert passed_pcc and passed_atol, err_msg


def _verify_onnx_module(
    filename,
    inputs,
    input_data_types,
    required_pcc,
    required_atol,
    input_range,
    input_range_int,
    compiler_config,
    do_assert,
):

    sess = InferenceSession(filename)
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

    inputs_dict = {
        nodearg.name: input.numpy().astype(np.float32)
        if input.dtype == torch.bfloat16
        else input.numpy()
        for nodearg, input in zip(sess.get_inputs(), inputs)
    }
    golden = sess.run(None, inputs_dict)

    for i in range(len(golden)):
        golden[i] = torch.tensor(golden[i])

    mod = onnx.load(filename)
    binary = compile_onnx(mod)

    ret = tt_mlir.run(inputs, binary)
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

    passed_pcc, passed_atol, msg, err_msg, _, _ = verify_against_golden(
        golden, ret, required_pcc, required_atol=required_atol
    )
    print(msg)
    if do_assert:
        assert passed_pcc and passed_atol, err_msg


def verify_module(
    mod,
    inputs=None,
    input_shapes=None,
    input_data_types=None,
    required_pcc=0.99,
    required_atol=1e-2,
    input_range=(-0.5, 0.5),
    input_range_int=(0, 1000),
    compiler_config=None,
    do_assert=True,
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
        )
    elif isinstance(mod, str) and mod.endswith(".onnx"):
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
