# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import os
import tt_mlir
import sys

from tt_torch.dynamo.torch_backend import (
    TorchFXExecutor,
    import_graph,
)
from tt_torch.dynamo.shlo_backend import (
    StableHLOExecutor,
    parse_module_from_str,
)
from tt_torch.dynamo.passes import pass_pipeline
from tt_torch.tools.utils import (
    CompilerConfig,
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


def _torch_backend(gm: torch.fx.GraphModule, example_inputs, compiler_config):
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


def torch_to_shlo(gm: torch.fx.GraphModule, example_inputs, compiler_config):
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


def shlo_to_flatbuffer(module, compiler_config):
    breakpoint()
    if compiler_config.profile_ops:
        compiler_config.set_stablehlo_mlir_module(module.operation.get_asm())
    if compiler_config.compile_depth == CompileDepth.STABLEHLO:
        return executor

    ttir = tt_mlir.compile_stable_hlo_to_ttir(module.operation.get_asm())
    dump_intermediates = os.environ.get("TT_TORCH_IR_LOG_LEVEL")
    if dump_intermediates:
        print("TTIR module", file=sys.stderr)
        print(ttir, file=sys.stderr)

    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
    if dump_intermediates:
        print("TTNN module", file=sys.stderr)
        print(ttnn, file=sys.stderr)

    return binary


def _base_backend(gm_or_shlo, example_inputs, compiler_config):
    if isinstance(gm_or_shlo, torch.fx.GraphModule):
        shlo = torch_to_shlo()
    elif isinstance(gm_or_shlo, str):
        shlo = parse_module_from_str(gm_or_shlo)
    else:
        print("Compiler input not valid", file=sys.stderr)
        exit(1)
    binary = shlo_to_flatbuffer(shlo, compiler_config)
    new_inputs = ()
    type_conversion = {torch.bool: torch.bfloat16}
    for input in example_inputs:
        # Handle scalar inputs.
        if not hasattr(input, "dtype"):
            assert (
                type(input) is not bool
            ), "Conversion for scalar boolean is not supported."
            new_inputs = new_inputs + ((input),)
            continue

        # Apply type conversion if required.
        input_type = input.dtype
        if input_type in type_conversion.keys():
            new_inputs = new_inputs + ((input.to(dtype=type_conversion[input_type])),)
            continue

        # No conversion required.
        new_inputs = new_inputs + ((input),)

    example_inputs = new_inputs
    tt_mlir.run(example_inputs, binary)


def backend(gm_or_shlo, example_inputs, options=None):
    if options is None:
        options = CompilerConfig()
    if (
        options.compile_depth == CompileDepth.COMPILE_OP_BY_OP
        or options.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
    ):
        # run op-by-op
        if isinstance(gm_or_shlo, torch.fx.GraphModule):
            # run torch op-by-op
            return _torch_backend(gm_or_shlo, example_inputs, compiler_config=options)
        elif isinstance(gm_or_shlo, str):
            # run shlo op-by-op
            return _shlo_backend(gm_or_shlo, options)
        else:
            print("Compiler input not valid", file=sys.stderr)
            exit(1)

    return _base_backend(gm_or_shlo, example_inputs, compiler_config=options)


def generate_random_inputs(module_str):
    # Parse tensor shapes from the module string
    import re

    tensor_shapes = re.findall(r"tensor<([\dx]+)xf32>", module_str)

    inputs = []
    for shape_str in tensor_shapes:
        shape = [int(dim) for dim in shape_str.split("x")]
        inputs.append(torch.randn(shape, dtype=torch.float32))

    return inputs


# Usage
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
example_inputs = generate_random_inputs(MODULE_STRING)
breakpoint()
compiler_config = CompilerConfig()
compiler_config.compile_depth = CompileDepth.EXECUTE
backend(MODULE_STRING, example_inputs, compiler_config)
