# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest
import os
import subprocess

import tt_torch
from tt_torch.tools.verify import verify_module
from tt_torch.tools.utils import CompilerConfig
import onnx
from tt_torch.onnx_compile import compile_onnx

from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.tools.onnx_model_utils import (
    make_dim_param_fixed,
    make_input_shape_fixed,
    fix_output_shapes,
)

shape_prop_nodes = ["QLinearAdd", "QLinearGlobalAveragePool"]


def shape_prop_node(node, input_shapes):
    if node.op_type == "QLinearAdd":
        # the first two input shapes need to be the same
        assert len(input_shapes) == 1 or input_shapes[0] == input_shapes[1]
        return input_shapes[0]
    elif node.op_type == "QLinearGlobalAveragePool":
        for attr in node.attribute:
            if attr.name == "channels_last":
                channels_last = True if attr.i == 1 else False
        if channels_last:
            N, H, W, C = input_shapes[0]
            return [N, 1, 1, C]
        else:
            N, C, H, W = input_shapes[0]
            return [N, C, 1, 1]
        assert False


def shape_from_value_info(value_info):
    shape = value_info.type.tensor_type.shape
    shape = [dim.dim_value for dim in shape.dim]
    shape_known = not any(dim == 0 or dim == -1 for dim in shape)
    return shape, shape_known


def resolve_shapes(mod):
    shapes_resolved = False
    while not shapes_resolved:
        shapes_resolved = True
        shapes = {}

        mod = onnx.shape_inference.infer_shapes(mod)

        for value_info in mod.graph.value_info:
            shapes[value_info.name] = value_info

        for index, node in enumerate(mod.graph.node):
            if node.op_type in shape_prop_nodes:
                assert node.input[0] in shapes
                shape_known = True
                for output in node.output:
                    if output not in shapes:
                        shape_known = False
                    else:
                        shape, shape_known = shape_from_value_info(shapes[output])
                    if not shape_known:
                        break

                if shape_known:
                    print(f"Shape known: {node.name}")
                    continue

                input_shapes = []
                for inp in node.input:
                    if inp in shapes:
                        shape, _ = shape_from_value_info(shapes[inp])
                        input_shapes.append(shape)

                output_shape = shape_prop_node(node, input_shapes)
                new_value_info = onnx.helper.make_tensor_value_info(
                    output,
                    shapes[node.input[0]].type.tensor_type.elem_type,
                    output_shape,
                )
                mod.graph.value_info.append(new_value_info)
                shapes_resolved = False
                break

    return mod


def test_resnet50():
    # download file if it doesn't exist
    if not os.path.exists("resnet50-v1-12-int8.onnx"):
        subprocess.run(
            [
                "wget",
                "-O",
                "resnet50-v1-12-int8.onnx",
                "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v1-12-int8.onnx",
            ]
        )
    mod = onnx.load("resnet50-v1-12-int8.onnx")
    for input in mod.graph.input:
        make_dim_param_fixed(mod.graph, input.name, 1)
        make_input_shape_fixed(mod.graph, input.name, [1, 3, 224, 224])
    mod = onnx.shape_inference.infer_shapes(mod)
    mod = resolve_shapes(mod)
    fix_output_shapes(mod)
    onnx.save(mod, "patched.onnx")
    compile_onnx(mod)
