# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import onnx
from onnxruntime import InferenceSession
import numpy as np
import tt_mlir
from tt_torch.onnx_compile import compile_onnx
from tt_torch.dynamo.backend import backend
from tt_torch.tools.utils import calculate_atol, calculate_pcc


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

    atol = calculate_atol(ret, golden)
    error = False
    if atol > required_atol:
        error = True

    if np.prod(golden.shape) != 1:
        ret = ret.to(torch.float32) if ret.dtype == torch.bfloat16 else ret
        golden = golden.to(torch.float32) if golden.dtype == torch.bfloat16 else golden

        pcc = calculate_pcc(ret, golden)

        if pcc < required_pcc:
            error = True

    if do_assert:
        assert not error, f"Error in verification: ATOL: {atol}, PCC: {pcc}"


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

    for golden_out, tt_out in zip(golden, ret):
        atol = calculate_atol(tt_out, golden_out)
        assert (
            do_assert and atol
        ) <= required_atol, f"ATOL too high: {atol} vs {required_atol}"

        if np.prod(golden_out.shape) == 1:
            return

        tt_out = tt_out.to(torch.float32) if tt_out.dtype == torch.bfloat16 else tt_out
        golden_out = (
            golden_out.to(torch.float32)
            if golden_out.dtype == torch.bfloat16
            else golden_out
        )

        pcc = calculate_pcc(tt_out, golden_out)

        assert (
            do_assert and pcc
        ) >= required_pcc, f"PCC too low: {pcc} vs {required_pcc}"


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
