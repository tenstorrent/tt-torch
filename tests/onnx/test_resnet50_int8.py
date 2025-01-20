# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest
import onnx


import tt_torch
from tt_torch.tools.verify import verify_module
from tt_torch.tools.utils import CompilerConfig
from tt_torch.onnx_compile import compile_onnx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.tools.onnx_model_utils import (
    make_dim_param_fixed,
    make_input_shape_fixed,
)


@pytest.mark.parametrize(
    "filename",
    [
        # "/localdev/achoudhury/tt-torch/tests/onnx/resnet50-v1-12-int8.onnx",
        # "/localdev/achoudhury/tt-torch/tests/onnx/mnist-12-int8.onnx"
        "/localdev/achoudhury/tt-torch/tests/onnx/mnist-1.onnx"
    ],
)
def test_generic(filename):
    mod = onnx.load(filename)
    print(onnx.helper.printable_graph(mod.graph))
    binary = compile_onnx(mod)


@pytest.mark.skip(reason="Need to fix the test")
def test_resnet():
    filename = "/localdev/achoudhury/tt-torch/tests/onnx/resnet50-v1-12-int8.onnx"
    # verify_module(filename)
    mod = onnx.load(filename)
    # print(mod.graph.input)
    print(onnx.helper.printable_graph(mod.graph))
    # mod = onnx.version_converter.convert_version(mod, 22)
    # mod = SymbolicShapeInference.infer_shapes(mod)
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
