# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import os
import tt_mlir

from tt_torch.dynamo.torch_backend import (
    TorchFXExecutor,
    import_graph,
)
from tt_torch.dynamo.shlo_backend import StableHLOExecutor
from tt_torch.dynamo.passes import pass_pipeline
from tt_torch.tools.utils import (
    CompilerConfig,
    FrontEnd,
    CompileDepth,
)


def lower_to_stable_hlo(module, op=None):
    run_pipeline_with_repro_report(
        module,
        f"builtin.module(torchdynamo-export-to-torch-backend-pipeline)",
        "Lowering TorchFX IR -> Torch Backend IR",
    )
    if op is not None:
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_TORCH_BACKEND_IR

    lower_mlir_module(False, OutputType.STABLEHLO, module)
    if op is not None:
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_STABLE_HLO


def _shlo_backend(module_str, options=None):
    if options is None:
        options = CompilerConfig()
    options.graph_type = "STABLEHLO"
    executor = StableHLOExecutor(module_str, compiler_config=options)
    executor()
    return executor


def _base_backend(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    # Apply environment overrides at start of compilation to allow overriding what was set in the test
    compiler_config.apply_environment_overrides()
    with torch.no_grad():
        gm, graph_constants = pass_pipeline(gm, example_inputs, compiler_config)
    executor = TorchFXExecutor(gm, graph_constants, compiler_config)
    if compiler_config.compile_depth in (
        CompileDepth.EXECUTE_OP_BY_OP,
        CompileDepth.COMPILE_OP_BY_OP,
        CompileDepth.TORCH_FX,
    ):
        return executor

    dump_intermediates = os.environ.get("TT_TORCH_IR_LOG_LEVEL")
    dump_intermediates = dump_intermediates and (
        dump_intermediates == "INFO" or dump_intermediates == "DEBUG"
    )

    module = import_graph(gm.graph)
    if dump_intermediates:
        print("Torch module", file=sys.stderr)
        module.dump()

    if compiler_config.profile_ops:
        compiler_config.set_torch_mlir_module(module.operation.get_asm())
    if compiler_config.compile_depth == CompileDepth.TORCH_MLIR:
        return executor

    lower_to_stable_hlo(module)
    if dump_intermediates:
        print("StableHLO module", file=sys.stderr)
        module.dump()

    if compiler_config.profile_ops:
        compiler_config.set_stablehlo_mlir_module(module.operation.get_asm())
    if compiler_config.compile_depth == CompileDepth.STABLEHLO:
        return executor

    ttir = tt_mlir.compile_stable_hlo_to_ttir(module.operation.get_asm())
    if dump_intermediates:
        print("TTIR module", file=sys.stderr)
        print(ttir, file=sys.stderr)

    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
    if dump_intermediates:
        print("TTNN module", file=sys.stderr)
        print(ttnn, file=sys.stderr)

    executor.set_binary(binary)
    return executor


def backend(gm_or_module, example_inputs, options=None):
    if options is None:
        options = CompilerConfig()
    if options.compiler_front_end == FrontEnd.STABLEHLO:
        if not isinstance(gm_or_module, str):
            exit(1)
        return _shlo_backend(gm_or_module, options)
    return _base_backend(gm_or_module, example_inputs, compiler_config=options)


MODULE_STRING = """
module {
  func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
    return %2 : tensor<1x128xf32>
  }
}
"""
compiler_config = CompilerConfig()
compiler_config.compiler_front_end = FrontEnd.STABLEHLO
compiler_config.compile_depth = CompileDepth.COMPILE_OP_BY_OP
backend(MODULE_STRING, None, compiler_config)
