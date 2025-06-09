# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from torch import nn
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # First operation
        a = x + y

        # Access tensor properties (likely to cause a graph break)
        shape_value = a.shape[0]

        # Control flow based on tensor data (another graph break source)
        if shape_value > 0:
            b = a * 2
        else:
            b = a * 3

        # Operations after potential graph breaks
        c = torch.nn.functional.relu(b)
        d = c * 2
        e = d * 3
        f = torch.nn.functional.gelu(e)

        return f


class TensorMetadataValidationTester(ModelTester):
    def _load_model(self):
        return ToyModel()

    def _load_inputs(self):
        # Create tensor inputs for the model
        # Don't randomize:
        x = torch.randn(4, 5)
        y = torch.randn(4, 5)
        return [x, y]


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_toy(record_property, mode, op_by_op):
    model_name = "ToyModel"

    cc = CompilerConfig()
    cc.enable_consteval = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO
        elif op_by_op == OpByOpBackend.TORCH:
            cc.op_by_op_backend = OpByOpBackend.TORCH

    tester = TensorMetadataValidationTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()

    print(f"Model test completed with results: {results}")

    tester.finalize()
