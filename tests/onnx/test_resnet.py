# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest

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


def test_resnet50():
    ## Download via wget https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v1-12-int8.onnx
    mod = onnx.load("resnet50-v1-12-int8.onnx")
    for input in mod.graph.input:
        make_dim_param_fixed(mod.graph, input.name, 1)
        make_input_shape_fixed(mod.graph, input.name, [1, 3, 224, 224])

    mod = onnx.shape_inference.infer_shapes(mod)
    mod = SymbolicShapeInference.infer_shapes(
        mod, auto_merge=True, guess_output_rank=True, verbose=1
    )
    shapes_resolved = False
    while not shapes_resolved:
        shapes_resolved = True
        intermediates = {}

        mod = onnx.shape_inference.infer_shapes(mod)
        for value_info in mod.graph.value_info:
            intermediates[value_info.name] = value_info

        for node, value_info in zip(mod.graph.node, mod.graph.value_info):
            if node.op_type == "QLinearAdd":
                assert node.input[0] in intermediates
                shape_known = True
                for output in node.output:
                    shape = intermediates[output].type.tensor_type.shape
                    shape = [dim.dim_value for dim in shape.dim]
                    shape_known = shape_known and not all(
                        [dim == 0 or dim == -1 for dim in shape]
                    )
                if shape_known:
                    print(f"Shape known: {node.name}")
                    continue

                shape = intermediates[node.input[0]].type.tensor_type.shape
                shape = [dim.dim_value for dim in shape.dim]
                new_value_info = onnx.helper.make_tensor_value_info(
                    output,
                    intermediates[node.input[0]].type.tensor_type.elem_type,
                    shape,
                )
                mod.graph.value_info.append(new_value_info)
                shapes_resolved = False
                break
    onnx.save(mod, "patched.onnx")
    breakpoint()
    compile_onnx(mod)
