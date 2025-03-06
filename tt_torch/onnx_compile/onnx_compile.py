# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import onnx
from torch_mlir.extras import onnx_importer
import tt_mlir
from torch_mlir.ir import Context
from torch_mlir.dialects import torch as torch_dialect
import os
import sys

from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
    lower_mlir_module,
)

from tt_torch.tools.utils import (
    OpByOpBackend,
    CompilerConfig,
    CompileDepth,
)

from tt_torch.dynamo.shlo_backend import StablehloExecutor
from tt_torch.dynamo.executor import Executor
from tt_torch.dynamo.backend import dump_module


def onnx_to_stablehlo(module: onnx.ModelProto, compiler_config):
    # Infer onnx shapes incase that information is missing
    module = onnx.shape_inference.infer_shapes(module)

    context = Context()
    torch_dialect.register_dialect(context)
    module_info = onnx_importer.ModelInfo(module)
    module = module_info.create_module(context=context).operation
    imp = onnx_importer.NodeImporter.define_function(module_info.main_graph, module)
    imp.import_all()

    dump_module(module=module, name="ONNX module", compiler_config=compiler_config)

    run_pipeline_with_repro_report(
        module,
        "builtin.module(torch-onnx-to-torch-backend-pipeline)",
        "Lowering Torch Onnx IR -> Torch Backend IR",
    )

    dump_module(
        module=module, name="Torch Backend module", compiler_config=compiler_config
    )

    lower_mlir_module(False, OutputType.STABLEHLO, module)

    dump_module(module=module, name="StableHLO module", compiler_config=compiler_config)
    return module


def compile_onnx(module: onnx.ModelProto, example_inputs, options=None):

    if options is None:
        options = CompilerConfig()
    options.op_by_op_backend = OpByOpBackend.STABLEHLO
    options.typecast_inputs = False
    options.apply_environment_overrides()
    if (
        options.compile_depth == CompileDepth.COMPILE_OP_BY_OP
        or options.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
    ):
        module = onnx_to_stablehlo(module, options)
        executor = StablehloExecutor(module=module, compiler_config=options)
        return executor(*example_inputs)
    elif options.compile_depth == CompileDepth.EXECUTE:
        module = onnx_to_stablehlo(module, options)
        executor = Executor(gm=None, graph_constants=None, compiler_config=options)
        ttir = tt_mlir.compile_stable_hlo_to_ttir(
            module.operation.get_asm(enable_debug_info=True)
        )
        dump_module(module=ttir, name="TTIR module", compiler_config=options)
        binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
        dump_module(module=ttnn, name="TTNN module", compiler_config=options)
        executor.set_binary(binary)
        return executor(*example_inputs)
