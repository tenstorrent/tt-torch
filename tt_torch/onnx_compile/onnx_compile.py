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


class OnnxExecutor:
    def __init__(self, model_proto: onnx.ModelProto):
        self.model_proto = model_proto
        self.binary = None
        self.module = None

    def set_module(self, module):
        self.module = module

    def set_model_proto(self, module: onnx.ModelProto):
        self.module = module

    def set_binary(self, binary):
        self.binary = binary

    def __call__(self, inputs):
        if self.binary is None:
            sess = ort.InferenceSession(self.model_proto.SerializeToString())
            outputs = sess.run(
                None,
                {
                    nodearg.name: inp.numpy()
                    if inp.dtype != torch.bfloat16
                    else inp.float().numpy()
                    for nodearg, inp in zip(sess.get_inputs(), inputs)
                },
            )
            return outputs

        return tt_mlir.run(inputs, self.binary)


def compile_onnx(module: onnx.ModelProto, compiler_config: CompilerConfig):
    # Infer onnx shapes incase that information is missing
    executor = OnnxExecutor(module)
    module = onnx.shape_inference.infer_shapes(module)
    executor.set_model_proto(module)

    context = Context()
    torch_dialect.register_dialect(context)
    module_info = onnx_importer.ModelInfo(module)
    module = module_info.create_module(context=context).operation
    imp = onnx_importer.NodeImporter.define_function(module_info.main_graph, module)
    imp.import_all()

    dump_intermediates = os.environ.get("TT_TORCH_IR_LOG_LEVEL")
    dump_info = False
    dump_debug = False
    if dump_intermediates:
        dump_debug = dump_intermediates == "DEBUG"
        dump_info = dump_debug or dump_intermediates == "INFO"

    # Setting large_elements_limit to 0 so the console does not get flooded with the data of large tensors
    if dump_info:
        print("ONNX module", file=sys.stderr)
        module.print(large_elements_limit=0)

    executor.set_module(module)
    if compiler_config.compile_depth == CompileDepth.TORCH_ONNX_IR:
        return executor

    run_pipeline_with_repro_report(
        module,
        "builtin.module(torch-onnx-to-torch-backend-pipeline)",
        "Lowering Torch Onnx IR -> Torch Backend IR",
    )

    if dump_info:
        print("Torch Backend module", file=sys.stderr)
        module.print(large_elements_limit=0)
        executor.set_module(module)

    if compiler_config.compile_depth == CompileDepth.TORCH_BACKEND_IR:
        return executor

    lower_mlir_module(False, OutputType.STABLEHLO, module)
    executor.set_module(module)

    if dump_info:
        print("StableHLO module", file=sys.stderr)
        module.print(large_elements_limit=0)

    executor.set_module(module)
    if compiler_config.compile_depth == CompileDepth.STABLEHLO:
        return executor

    # Need to set enable_debug_info=True to get the location information for the ops in the asm string
    ttir = tt_mlir.compile_stable_hlo_to_ttir(
        module.operation.get_asm(enable_debug_info=True)
    )

    executor.set_binary(ttir)
    if dump_info:
        print("TTIR module", file=sys.stderr)
        print(ttir, file=sys.stderr)

    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
    if dump_info:
        print("TTNN module", file=sys.stderr)
        print(ttnn, file=sys.stderr)

    executor.set_binary(binary)
    executor.set_module(module)
    return executor
