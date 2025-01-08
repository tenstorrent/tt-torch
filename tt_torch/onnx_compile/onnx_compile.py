# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import onnx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from torch_mlir.extras import onnx_importer
import tt_mlir
from torch_mlir.ir import Context
from torch_mlir.dialects import torch as torch_dialect
from onnx import version_converter

from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
    lower_mlir_module,
)


def compile_onnx(module: onnx.ModelProto):
    assert isinstance(module, onnx.ModelProto), "Expected onnx.ModelProto object"

    # Infer onnx shapes in case that information is missing
    module = version_converter.convert_version(module, 13)
    try:
        module = SymbolicShapeInference.infer_shapes(module)
    except Exception as e:
        raise Exception(
            f"Failed to infer shapes of onnx.ModelProto, it is possible this issue is caused by dynamic input shapes. \n Exception raised by SymbolicShapeInference.infer_shape: {e}"
        )

    onnx.checker.check_model(module)
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
