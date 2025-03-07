# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import onnx
import torch
import onnxruntime as ort
from torch_mlir.extras import onnx_importer
import tt_mlir
from torch_mlir.ir import Context
from torch_mlir.dialects import torch as torch_dialect
import os
import sys

from tt_torch.tools.utils import CompilerConfig, CompileDepth

from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
    lower_mlir_module,
)

from torch_mlir.ir import Module

from tt_torch.tools.utils import (
    OpByOpBackend,
    CompilerConfig,
    CompileDepth,
)

from tt_torch.dynamo.shlo_backend import StablehloExecutor
from tt_torch.dynamo.executor import Executor
from tt_torch.dynamo.backend import (
    dump_module,
    torch_backend_ir_to_stablehlo,
    stablehlo_to_ttir,
    ttir_to_ttnn_and_binary,
)


class OnnxExecutor:
    def __init__(self, model_proto: onnx.ModelProto):
        self.model_proto = model_proto
        self.binary = None
        self.sess = None

    def set_binary(self, binary):
        self.binary = binary

    def __call__(self, *inputs):
        if self.binary is None:
            # Only want to load the model proto into one inference session
            # since models can be big
            if self.sess is None:
                self.sess = ort.InferenceSession(self.model_proto.SerializeToString())
            outputs = self.sess.run(
                None,
                {
                    nodearg.name: inp.numpy()
                    if inp.dtype != torch.bfloat16
                    else inp.float().numpy()
                    for nodearg, inp in zip(self.sess.get_inputs(), inputs)
                },
            )
            return outputs

        return tt_mlir.run(inputs, self.binary)


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


def torch_onnx_to_torch_backend_ir(
    onnx_ir_module: Module, compiler_config: CompilerConfig
):
    run_pipeline_with_repro_report(
        onnx_ir_module,
        "builtin.module(torch-onnx-to-torch-backend-pipeline)",
        "Lowering Torch Onnx IR -> Torch Backend IR",
    )

    dump_module(
        module=onnx_ir_module,
        name="Torch Backend module",
        compiler_config=compiler_config,
    )
    return onnx_ir_module


def compile_onnx(model_proto: onnx.ModelProto, options=None):
    if options is None:
        options = CompilerConfig()

    assert isinstance(
        options, CompilerConfig
    ), "options must be an instance of CompilerConfig"

    options.op_by_op_backend = OpByOpBackend.STABLEHLO
    options.typecast_inputs = False
    options.apply_environment_overrides()
    if (
        options.compile_depth == CompileDepth.COMPILE_OP_BY_OP
        or options.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
    ):
        model_proto = onnx.shape_inference.infer_shapes(model_proto)
        module = generate_torch_onnx_ir(model_proto, options)
        module = torch_onnx_to_torch_backend_ir(module, options)
        module = torch_backend_ir_to_stablehlo(module, options)
        executor = StablehloExecutor(module=module, compiler_config=options)
        return executor
    else:
        model_proto = onnx.shape_inference.infer_shapes(model_proto)
        executor = OnnxExecutor(model_proto)

        module = generate_torch_onnx_ir(model_proto, options)
        if options.compile_depth == CompileDepth.TORCH_ONNX_IR:
            return executor

        module = torch_onnx_to_torch_backend_ir(module, options)
        if options.profile_ops:
            options.set_torch_mlir_module(module.operation.get_asm())
        if options.compile_depth == CompileDepth.TORCH_BACKEND_IR:
            return executor

        module = torch_backend_ir_to_stablehlo(module, options)
        if options.profile_ops:
            options.set_stablehlo_mlir_module(module.operation.get_asm())
        if options.compile_depth == CompileDepth.STABLEHLO:
            return executor

        module = stablehlo_to_ttir(module, options)
        if options.compile_depth == CompileDepth.TTIR_DIALECT:
            return executor

        binary, ttnn = ttir_to_ttnn_and_binary(module, options)
        if options.compile_depth == CompileDepth.TTNN_DIALECT:
            return executor

        executor.set_binary(binary)
        return executor
