# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import onnx
import torch
import onnxruntime as ort

# from torch_mlir.extras import onnx_importer
# import tt_mlir

# from torch_mlir.ir import Context
# from torch_mlir.dialects import torch as torch_dialect
import os
import sys

from tt_torch.tools.utils import CompilerConfig, CompileDepth, tt_torch_error_message

# from torch_mlir.compiler_utils import (
#     OutputType,
#     run_pipeline_with_repro_report,
#     lower_mlir_module,
# )

# from torch_mlir.ir import Module

from tt_torch.tools.utils import (
    OpByOpBackend,
    CompilerConfig,
    CompileDepth,
    MultiChipGraph,
)

# from tt_torch.dynamo.shlo_backend import StablehloExecutor
# from tt_torch.dynamo.executor import OnnxExecutor
# from tt_torch.dynamo.backend import dump_module, shlo_to_flatbuffer


def generate_torch_onnx_ir(module: onnx.ModelProto, compiler_config: CompilerConfig):
    context = Context()
    torch_dialect.register_dialect(context)
    module_info = onnx_importer.ModelInfo(module)
    module = module_info.create_module(context=context).operation
    imp = onnx_importer.NodeImporter.define_function(module_info.main_graph, module)
    imp.import_all()

    dump_module(
        module=module, name="Torch Onnx module", compiler_config=compiler_config
    )
    return module


def torch_onnx_to_torch_backend_ir(module, compiler_config: CompilerConfig):
    run_pipeline_with_repro_report(
        module,
        "builtin.module(torch-onnx-to-torch-backend-pipeline)",
        "Lowering Torch Onnx IR -> Torch Backend IR",
    )

    dump_module(
        module=module,
        name="Torch Backend module",
        compiler_config=compiler_config,
    )
    return module


def torch_backend_ir_to_stablehlo(module, compiler_config: CompilerConfig):
    lower_mlir_module(False, OutputType.STABLEHLO, module)
    dump_module(module=module, name="StableHLO module", compiler_config=compiler_config)
    return module


@tt_torch_error_message
def compile_onnx(model_proto: onnx.ModelProto, compiler_config: CompilerConfig = None):
    if compiler_config is None:
        compiler_config = CompilerConfig()

    assert isinstance(
        compiler_config, CompilerConfig
    ), "compiler_config must be an instance of CompilerConfig"

    compiler_config.op_by_op_backend = OpByOpBackend.STABLEHLO
    compiler_config.typecast_inputs = False
    compiler_config.apply_environment_overrides()
    model_proto = onnx.shape_inference.infer_shapes(model_proto)
    onnx.save(model_proto, "model.onnx")
    shape_inference.quant_pre_process(
        input_model_path="model.onnx",
        output_model_path="processed_model.onnx",
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
    )
    model_proto = onnx.load("processed_model.onnx")
    if (
        compiler_config.compile_depth == CompileDepth.COMPILE_OP_BY_OP
        or compiler_config.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
    ):
        module = generate_torch_onnx_ir(model_proto, compiler_config)
        module = torch_onnx_to_torch_backend_ir(module, compiler_config)
        module = torch_backend_ir_to_stablehlo(module, compiler_config)
        executor = StablehloExecutor(module=module, compiler_config=compiler_config)
        executor.add_onnx_model_proto(model_proto)
        return executor
    else:
        executor = OnnxExecutor(model_proto)

        module = generate_torch_onnx_ir(model_proto, compiler_config)

        module = torch_onnx_to_torch_backend_ir(module, compiler_config)
        if compiler_config.profile_ops:
            compiler_config.set_torch_mlir_module(module.operation.get_asm())

        module = torch_backend_ir_to_stablehlo(module, compiler_config)
        if compiler_config.profile_ops:
            compiler_config.set_stablehlo_mlir_module(module.operation.get_asm())
        if compiler_config.compile_depth == CompileDepth.STABLEHLO:
            return executor
        # TODO: Add consteval support for onnx https://github.com/tenstorrent/tt-torch/issues/703
        binary = shlo_to_flatbuffer(
            executor, executor.system_desc_paths[0], module, compiler_config, 0, 0
        )
        executor.set_binary(binary)
        return executor
