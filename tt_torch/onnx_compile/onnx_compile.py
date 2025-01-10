# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import onnx
from torch_mlir.extras import onnx_importer
import tt_mlir
from torch_mlir.ir import Context
from torch_mlir.dialects import torch as torch_dialect

from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
    lower_mlir_module,
)


def compile_onnx(module: onnx.ModelProto, inputs):
    # Infer onnx shapes incase that information is missing
    for input_node in module.graph.input:
        assert input_node.name in inputs, f"Input {input_node.name} not provided"

        for i, dim in enumerate(input_node.type.tensor_type.shape.dim):
            dim.Clear()
            dim.dim_value = inputs[input_node.name].shape[i]

    module = onnx.version_converter.convert_version(module, 17)
    module = SymbolicShapeInference.infer_shapes(module)

    context = Context()
    torch_dialect.register_dialect(context)
    module_info = onnx_importer.ModelInfo(module)
    module = module_info.create_module(context=context).operation
    imp = onnx_importer.NodeImporter.define_function(module_info.main_graph, module)
    imp.import_all()

    run_pipeline_with_repro_report(
        module,
        "builtin.module(torch-onnx-to-torch-backend-pipeline)",
        "Lowering Torch Onnx IR -> Torch Backend IR",
    )
    lower_mlir_module(False, OutputType.STABLEHLO, module)
    return tt_mlir.compile(module.operation.get_asm())
