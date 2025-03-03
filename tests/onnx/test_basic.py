# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest
import onnx
from onnx import helper, TensorProto

import tt_torch
from tt_torch.tools.verify import verify_module
from tt_torch.tools.utils import CompilerConfig


def test_add():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.add(x, y)

    torch_model = Basic()
    torch_input = (torch.randn(256, 256), torch.randn(256, 256))
    torch.onnx.export(torch_model, torch_input, "add.onnx")
    cc = CompilerConfig()
    cc.compile_depth = tt_torch.tools.utils.CompileDepth.EXECUTE_OP_BY_OP
    cc.op_by_op_backend = tt_torch.tools.utils.OpByOpBackend.STABLEHLO
    verify_module("add.onnx", compiler_config=cc)


def test_abs():
    def create_abs_model():
        input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])
        output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])
        abs_node = helper.make_node(
            "Abs", inputs=["input"], outputs=["output"]  # ONNX Abs operation
        )
        graph = helper.make_graph([abs_node], "AbsModel", [input_info], [output_info])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
        return model

    onnx.save(create_abs_model(), "abs.onnx")
    cc = CompilerConfig()
    cc.compile_depth = tt_torch.tools.utils.CompileDepth.EXECUTE_OP_BY_OP
    cc.op_by_op_backend = tt_torch.tools.utils.OpByOpBackend.STABLEHLO
    verify_module("abs.onnx", compiler_config=cc)
