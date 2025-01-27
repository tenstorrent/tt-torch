# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest
import onnx
from difflib import ndiff


import tt_torch
from tt_torch.tools.verify import verify_module
from tt_torch.tools.utils import CompilerConfig
from tt_torch.onnx_compile import compile_onnx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.tools.onnx_model_utils import (
    make_dim_param_fixed,
    make_input_shape_fixed,
    fix_output_shapes,
)
from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    quant_pre_process,
)


@pytest.mark.parametrize(
    "filename",
    [
        # "/localdev/achoudhury/tt-torch/tests/onnx/resnet50-v1-12-int8.onnx",
        # "/localdev/achoudhury/tt-torch/tests/onnx/mnist-12-int8.onnx"
        # "/localdev/achoudhury/tt-torch/tests/onnx/mnist-1.onnx"
        # "/localdev/achoudhury/tt-torch/tests/onnx/mobilenetv2-12-int8.onnx"
        # # tiny yolo is too large to print IR dumps of
        # "/localdev/achoudhury/tt-torch/tests/onnx/tinyyolov2-7.onnx"
        "/localdev/achoudhury/tt-torch/tests/onnx/distilbert.onnx"
        # "/localdev/achoudhury/tt-torch/tests/onnx/mnist_custom.onnx"
    ],
)
def test_generic(filename):
    mod = onnx.load(filename)
    # graph_before=onnx.helper.printable_graph(mod.graph)
    # print(graph_before)
    # print(len(mod.graph.input[0].type.tensor_type.shape.dim))
    # print(mod.graph.input)
    # print("----------------")
    # print(mod.graph.node)
    # for node in mod.graph.node:
    #     # if node.op_type == "Constant":
    #         print(node)
    #         print(node.attribute[0].t)
    #         print(node.attribute[0].type)

    mod = onnx.version_converter.convert_version(mod, 17)
    for input in mod.graph.input:
        make_dim_param_fixed(mod.graph, input.name, 1)
        make_input_shape_fixed(mod.graph, input.name, [32, 32])
        # make_input_shape_fixed(mod.graph, input.name, [32] * len(input.type.tensor_type.shape.dim))
    fix_output_shapes(mod)
    mod = SymbolicShapeInference.infer_shapes(mod, verbose=1)

    quntized_filename = filename[:-5] + "-quantized.onnx"

    # quant_pre_process(mod, quntized_filename)
    # mod = onnx.load(quntized_filename)

    quantize_dynamic(mod, quntized_filename)
    # quantize_static(mod, quntized_filename)
    mod = onnx.load(quntized_filename)

    mod = onnx.version_converter.convert_version(mod, 17)
    for input in mod.graph.input:
        make_dim_param_fixed(mod.graph, input.name, 1)
        make_input_shape_fixed(mod.graph, input.name, [32, 32])
        # make_input_shape_fixed(mod.graph, input.name, [32] * len(input.type.tensor_type.shape.dim))
    fix_output_shapes(mod)
    mod = SymbolicShapeInference.infer_shapes(mod, verbose=1)

    # graph_after=onnx.helper.printable_graph(mod.graph)
    # print(graph_after)
    # diff = "".join((line for line in ndiff(graph_before.splitlines(1), graph_after.splitlines(1)) if line[0] != " "))
    # print("brata diff", diff)
    binary = compile_onnx(mod)


@pytest.mark.skip(reason="Need to fix the test")
def test_resnet():
    filename = "/localdev/achoudhury/tt-torch/tests/onnx/resnet50-v1-12-int8.onnx"
    # verify_module(filename)
    mod = onnx.load(filename)
    # print(mod.graph.input)
    print(onnx.helper.printable_graph(mod.graph))
    # mod = onnx.version_converter.convert_version(mod, 22)
    mod = SymbolicShapeInference.infer_shapes(mod)
    # mod = SymbolicShapeInference.infer_shapes(mod, auto_merge=True, guess_output_rank=True, verbose = 1)

    # make_dim_param_fixed(mod.graph, "data", 1)
    # make_input_shape_fixed(mod.graph, "data", [1, 3, 224, 224])
    # print(mod.graph.input)
    node = mod.graph.node[0]
    print(node)
    new_graph = onnx.helper.make_graph(
        nodes=[node],
        name=node.name,  # mod.graph.name,
        inputs=node.input,  # mod.graph.input,
        outputs=node.output,  # mod.graph.output,
        # initializer = mod.graph.initializer
    )
    new_model = onnx.helper.make_model(new_graph, producer_name="onnx-tensorrt")
    binary = compile_onnx(new_model)
