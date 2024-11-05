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


def _verify_torch_module(
    mod,
    input_shapes,
    input_data_types,
    required_pcc,
    required_atol,
    input_range,
    compiler_config,
    do_assert,
):
    tt_mod = torch.compile(mod, backend=backend, options=compiler_config)

    if all([dtype.is_floating_point for dtype in input_data_types]):
        low, high = input_range
        # Uniformly distribute random numbers within the input_range
        inputs = [(low - high) * torch.rand(shape) + high for shape in input_shapes]
    else:
        inputs = [
            torch.randint(0, 1000, shape, dtype=torch.int32) for shape in input_shapes
        ]
    ret = tt_mod(*inputs)
    golden = mod(*inputs)

    atol = torch.max(torch.abs(golden - ret)).item()
    if do_assert:
        assert atol <= required_atol, f"ATOL too high: {atol} vs {required_atol}"

    if np.prod(golden.shape) == 1:
        return
    pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(ret).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
        )
    )
    if do_assert:
        assert pcc >= required_pcc, f"PCC too low: {pcc} vs {required_pcc}"


def _verify_onnx_module(
    filename,
    input_data_types,
    required_pcc,
    required_atol,
    input_range,
    compiler_config,
    do_assert,
):

    sess = InferenceSession(filename)
    input_shapes = [nodearg.shape for nodearg in sess.get_inputs()]

    if all([dtype.is_floating_point for dtype in input_data_types]):
        low, high = input_range
        # Uniformly distribute random numbers within the input_range
        inputs = [(low - high) * torch.rand(shape) + high for shape in input_shapes]
    else:
        inputs = [
            torch.randint(0, 1000, shape, dtype=torch.int32) for shape in input_shapes
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
        atol = torch.max(torch.abs(golden_out - tt_out)).item()
        assert (
            do_assert and atol
        ) <= required_atol, f"ATOL too high: {atol} vs {required_atol}"

        if np.prod(golden_out.shape) == 1:
            return
        pcc = np.min(
            np.ma.corrcoef(
                np.ma.masked_invalid(torch.squeeze(tt_out).detach().numpy()).flatten(),
                np.ma.masked_invalid(
                    torch.squeeze(golden_out).detach().numpy()
                ).flatten(),
            )
        )
        assert (
            do_assert and pcc
        ) >= required_pcc, f"PCC too low: {pcc} vs {required_pcc}"


def verify_module(
    mod,
    input_shapes=None,
    input_data_types=[torch.float32],
    required_pcc=0.99,
    required_atol=1e-2,
    input_range=(-0.5, 0.5),
    compiler_config=None,
    do_assert=True,
):
    if isinstance(mod, torch.nn.Module):
        assert (
            input_shapes is not None
        ), "Verifying a torch module requires that you provide input_shapes"
        _verify_torch_module(
            mod,
            input_shapes,
            input_data_types,
            required_pcc,
            required_atol,
            input_range,
            compiler_config,
            do_assert,
        )
    elif isinstance(mod, str) and mod.endswith(".onnx"):
        assert (
            input_shapes is None
        ), "When verifying an ONNX module, input_shapes must be None as they are inferred from the ONNX model"
        _verify_onnx_module(
            mod,
            input_data_types,
            required_pcc,
            required_atol,
            input_range,
            compiler_config,
            do_assert,
        )
    else:
        raise ValueError("Invalid module type")
