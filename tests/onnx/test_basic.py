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


def test_add():
    def create_add_model():
        input1_info = helper.make_tensor_value_info(
            "input1", TensorProto.FLOAT, [256, 256]
        )
        input2_info = helper.make_tensor_value_info(
            "input2", TensorProto.FLOAT, [256, 256]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [256, 256]
        )
        add_node = helper.make_node(
            "Add", inputs=["input1", "input2"], outputs=["output"]
        )
        graph = helper.make_graph(
            [add_node], "AddModel", [input1_info, input2_info], [output_info]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
        return model

    onnx.save(create_add_model(), "add.onnx")
    cc = tt_torch.tools.utils.CompilerConfig()
    cc.compile_depth = tt_torch.tools.utils.CompileDepth.EXECUTE_OP_BY_OP
    cc.op_by_op_backend = tt_torch.tools.utils.OpByOpBackend.STABLEHLO
    verify_module("add.onnx", compiler_config=cc)


def test_concat_dim0():
    def create_concat_model():
        input1_info = helper.make_tensor_value_info(
            "input1", TensorProto.FLOAT, [32, 32]
        )
        input2_info = helper.make_tensor_value_info(
            "input2", TensorProto.FLOAT, [64, 32]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [96, 32]
        )
        concat_node = helper.make_node(
            "Concat", inputs=["input1", "input2"], outputs=["output"], axis=0
        )
        graph = helper.make_graph(
            [concat_node], "ConcatModel", [input1_info, input2_info], [output_info]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
        return model

    onnx.save(create_concat_model(), "concat_dim0.onnx")
    cc = tt_torch.tools.utils.CompilerConfig()
    cc.compile_depth = tt_torch.tools.utils.CompileDepth.EXECUTE_OP_BY_OP
    cc.op_by_op_backend = tt_torch.tools.utils.OpByOpBackend.STABLEHLO
    verify_module("concat_dim0.onnx", compiler_config=cc)


def test_concat_dim1():
    def create_concat_model():
        input1_info = helper.make_tensor_value_info(
            "input1", TensorProto.FLOAT, [32, 32]
        )
        input2_info = helper.make_tensor_value_info(
            "input2", TensorProto.FLOAT, [32, 64]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [32, 96]
        )
        concat_node = helper.make_node(
            "Concat", inputs=["input1", "input2"], outputs=["output"], axis=1
        )
        graph = helper.make_graph(
            [concat_node], "ConcatModel", [input1_info, input2_info], [output_info]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
        return model

    onnx.save(create_concat_model(), "concat_dim1.onnx")
    cc = tt_torch.tools.utils.CompilerConfig()
    cc.compile_depth = tt_torch.tools.utils.CompileDepth.EXECUTE_OP_BY_OP
    cc.op_by_op_backend = tt_torch.tools.utils.OpByOpBackend.STABLEHLO
    verify_module("concat_dim1.onnx", compiler_config=cc)


def test_concat_dim2():
    def create_concat_model():
        input1_info = helper.make_tensor_value_info(
            "input1", TensorProto.FLOAT, [32, 32, 32]
        )
        input2_info = helper.make_tensor_value_info(
            "input2", TensorProto.FLOAT, [32, 32, 64]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [32, 32, 96]
        )
        concat_node = helper.make_node(
            "Concat", inputs=["input1", "input2"], outputs=["output"], axis=2
        )
        graph = helper.make_graph(
            [concat_node], "ConcatModel", [input1_info, input2_info], [output_info]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
        return model

    onnx.save(create_concat_model(), "concat_dim2.onnx")
    cc = tt_torch.tools.utils.CompilerConfig()
    cc.compile_depth = tt_torch.tools.utils.CompileDepth.EXECUTE_OP_BY_OP
    cc.op_by_op_backend = tt_torch.tools.utils.OpByOpBackend.STABLEHLO
    verify_module("concat_dim2.onnx", compiler_config=cc)


def test_concat_dim3():
    def create_concat_model():
        input1_info = helper.make_tensor_value_info(
            "input1", TensorProto.FLOAT, [32, 32, 32, 32]
        )
        input2_info = helper.make_tensor_value_info(
            "input2", TensorProto.FLOAT, [32, 32, 32, 64]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [32, 32, 32, 96]
        )
        concat_node = helper.make_node(
            "Concat", inputs=["input1", "input2"], outputs=["output"], axis=3
        )
        graph = helper.make_graph(
            [concat_node], "ConcatModel", [input1_info, input2_info], [output_info]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
        return model

    onnx.save(create_concat_model(), "concat_dim3.onnx")
    cc = tt_torch.tools.utils.CompilerConfig()
    cc.compile_depth = tt_torch.tools.utils.CompileDepth.EXECUTE_OP_BY_OP
    cc.op_by_op_backend = tt_torch.tools.utils.OpByOpBackend.STABLEHLO
    verify_module("concat_dim3.onnx", compiler_config=cc)


@pytest.mark.parametrize(
    ("input_range", "input_shapes", "input_type"),
    [
        ((-0.5, 0.5), [(2, 2), (2, 2)], TensorProto.FLOAT),
        ((1, 10), [(32, 32), (32, 32)], TensorProto.FLOAT),
        ((1, 10), [(32, 32), (32, 32)], TensorProto.FLOAT),
    ],
)
def test_div(input_range, input_shapes, input_type):
    def create_div_model():
        input1_info = helper.make_tensor_value_info(
            "input1", input_type, input_shapes[0]
        )
        input2_info = helper.make_tensor_value_info(
            "input2", input_type, input_shapes[1]
        )
        output_info = helper.make_tensor_value_info(
            "output", input_type, input_shapes[0]
        )
        div_node = helper.make_node(
            "Div", inputs=["input1", "input2"], outputs=["output"]
        )
        graph = helper.make_graph(
            [div_node], "DivModel", [input1_info, input2_info], [output_info]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
        return model

    onnx.save(create_div_model(), "div.onnx")
    cc = tt_torch.tools.utils.CompilerConfig()
    cc.compile_depth = tt_torch.tools.utils.CompileDepth.EXECUTE_OP_BY_OP
    cc.op_by_op_backend = tt_torch.tools.utils.OpByOpBackend.STABLEHLO
    verify_module("div.onnx", compiler_config=cc)


def test_div_zero():
    def create_div_zero_model():
        input1_info = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [8])
        input2_info = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [8])
        output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [8])
        div_node = helper.make_node(
            "Div", inputs=["input1", "input2"], outputs=["output"]
        )
        graph = helper.make_graph(
            [div_node], "DivZeroModel", [input1_info, input2_info], [output_info]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
        return model

    onnx.save(create_div_zero_model(), "div_zero.onnx")
    cc = tt_torch.tools.utils.CompilerConfig()
    cc.compile_depth = tt_torch.tools.utils.CompileDepth.EXECUTE_OP_BY_OP
    cc.op_by_op_backend = tt_torch.tools.utils.OpByOpBackend.STABLEHLO
    verify_module("div_zero.onnx", compiler_config=cc)


def test_exp():
    def create_exp_model():
        input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 2])
        output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2])
        exp_node = helper.make_node("Exp", inputs=["input"], outputs=["output"])
        graph = helper.make_graph([exp_node], "ExpModel", [input_info], [output_info])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
        return model

    onnx.save(create_exp_model(), "exp.onnx")
    cc = tt_torch.tools.utils.CompilerConfig()
    cc.compile_depth = tt_torch.tools.utils.CompileDepth.EXECUTE_OP_BY_OP
    cc.op_by_op_backend = tt_torch.tools.utils.OpByOpBackend.STABLEHLO
    verify_module("exp.onnx", compiler_config=cc, required_atol=3e-2)


def test_linear_with_bias():
    def create_linear_with_bias_model():
        input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 32])
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [32, 32]
        )

        weights_info = helper.make_tensor_value_info(
            "weights", TensorProto.FLOAT, [32, 32]
        )
        bias_info = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [32])
        matmul_node = helper.make_node(
            "MatMul", inputs=["input", "weights"], outputs=["matmul_output"]
        )
        add_node = helper.make_node(
            "Add", inputs=["matmul_output", "bias"], outputs=["output"]
        )

        graph = helper.make_graph(
            [matmul_node, add_node],
            "LinearWithBiasModel",
            [input_info, weights_info, bias_info],
            [output_info],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
        return model

    onnx.save(create_linear_with_bias_model(), "linear_with_bias.onnx")
    cc = tt_torch.tools.utils.CompilerConfig()
    cc.compile_depth = tt_torch.tools.utils.CompileDepth.EXECUTE
    cc.op_by_op_backend = tt_torch.tools.utils.OpByOpBackend.STABLEHLO
    verify_module("linear_with_bias.onnx", compiler_config=cc)
