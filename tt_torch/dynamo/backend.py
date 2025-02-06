# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import os
import tt_mlir
import sys
import torch_mlir

from tt_torch.dynamo.torch_backend import (
    TorchExecutor,
    import_graph,
)
from tt_torch.dynamo.shlo_backend import (
    generate_random_inputs_for_shlo,
    parse_module_from_str,
    StablehloExecutor,
)
from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
    lower_mlir_module,
)
from tt_torch.dynamo.passes import pass_pipeline
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
)


def lower_to_stable_hlo(module, op=None, enable_ir_printing=False):
    run_pipeline_with_repro_report(
        module,
        f"builtin.module(torchdynamo-export-to-torch-backend-pipeline)",
        "Lowering TorchFX IR -> Torch Backend IR",
        enable_ir_printing,
    )
    if op is not None:
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_TORCH_BACKEND_IR

    lower_mlir_module(False, OutputType.STABLEHLO, module)
    if op is not None:
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_STABLE_HLO


def verify_golden_callback(binary, callback_context, op_context):
    # Using these parameters, we should be able to query information
    # about the op described by op_context, and its output. I.e. location:
    location = tt_mlir.get_op_loc_info(op_context)
    # ...

    # We will need to provide the bindings necesarry in this frontend.
    # Those bindings will interact with the runtime API


def _shlo_backend(shlo, example_inputs, compiler_config, gm=None, graph_constants=None):
    if isinstance(shlo, torch_mlir._mlir_libs._mlir.ir.Module):
        executor = StablehloExecutor(
            parsed_module=shlo, compiler_config=compiler_config
        )
    elif isinstance(shlo, str):
        executor = StablehloExecutor(module_str=shlo, compiler_config=compiler_config)
    else:
        print("Compiler input not valid", file=sys.stderr)
        exit(1)
    if gm is not None:
        executor.add_gm(gm, graph_constants)
    return executor


def _torch_backend(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    # Apply environment overrides at start of compilation to allow overriding what was set in the test
    compiler_config.apply_environment_overrides()
    with torch.no_grad():
        gm, graph_constants = pass_pipeline(gm, example_inputs, compiler_config)
    executor = TorchExecutor(gm, graph_constants, compiler_config)
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

    # Need to set enable_debug_info=True to get the location information for the ops in the asm string
    ttir = tt_mlir.compile_stable_hlo_to_ttir(
        module.operation.get_asm(enable_debug_info=True)
    )
    if dump_info:
        print("TTIR module", file=sys.stderr)
        print(ttir, file=sys.stderr)

    if compiler_config.enable_intermediate_verification:
        executor.register_intermediate_callback(verify_golden_callback)

    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
    if dump_info:
        print("TTNN module", file=sys.stderr)
        print(ttnn, file=sys.stderr)

    executor.set_binary(binary)
    return executor


def torch_to_shlo(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    # Apply environment overrides at start of compilation to allow overriding what was set in the test
    compiler_config.apply_environment_overrides()
    with torch.no_grad():
        gm, graph_constants = pass_pipeline(gm, example_inputs, compiler_config)
    executor = TorchExecutor(gm, graph_constants, compiler_config)
    dump_intermediates = os.environ.get("TT_TORCH_IR_LOG_LEVEL")
    dump_info = False
    dump_debug = False
    if dump_intermediates:
        dump_debug = dump_intermediates == "DEBUG"
        dump_info = dump_debug or dump_intermediates == "INFO"

    module = import_graph(gm.graph)
    if dump_info:
        print("Torch module", file=sys.stderr)
        module.dump()

    if compiler_config.profile_ops:
        compiler_config.set_torch_mlir_module(module.operation.get_asm())

    lower_to_stable_hlo(module, enable_ir_printing=dump_debug)
    if dump_info:
        print("StableHLO module", file=sys.stderr)
        module.dump()

    return module, executor, gm, graph_constants


def shlo_to_flatbuffer(module, compiler_config):
    dump_intermediates = os.environ.get("TT_TORCH_IR_LOG_LEVEL")
    dump_info = False
    dump_debug = False
    if dump_intermediates:
        dump_debug = dump_intermediates == "DEBUG"
        dump_info = dump_debug or dump_intermediates == "INFO"

    if compiler_config.profile_ops:
        compiler_config.set_stablehlo_mlir_module(module.operation.get_asm())

    ttir = tt_mlir.compile_stable_hlo_to_ttir(
        module.operation.get_asm(enable_debug_info=True)
    )
    if dump_info:
        print("TTIR module", file=sys.stderr)
        print(ttir, file=sys.stderr)

    if compiler_config.enable_intermediate_verification:
        executor.register_intermediate_callback(verify_golden_callback)

    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
    if dump_info:
        print("TTNN module", file=sys.stderr)
        print(ttnn, file=sys.stderr)

    return binary


def _base_backend(gm_or_shlo, example_inputs, compiler_config):
    # Called during EXECUTE
    # input is a torch graph
    if isinstance(gm_or_shlo, torch.fx.GraphModule):
        shlo, executor, gm, graph_constants = torch_to_shlo(
            gm_or_shlo, example_inputs, compiler_config
        )
    # input is a stablehlo string module
    elif isinstance(gm_or_shlo, str):
        shlo = parse_module_from_str(gm_or_shlo)
        executor = StablehloExecutor(
            parsed_module=shlo, compiler_config=compiler_config
        )
    else:
        print("Compiler input not valid", file=sys.stderr)
        exit(1)

    binary = shlo_to_flatbuffer(shlo, compiler_config)
    executor.set_binary(binary)
    return executor


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
            return _shlo_backend(
                shlo=gm_or_shlo, example_inputs=example_inputs, compiler_config=options
            )
        else:
            print("Compiler input not valid", file=sys.stderr)
            exit(1)

    if options.compile_depth == CompileDepth.EXECUTE:
        return _base_backend(gm_or_shlo, example_inputs, compiler_config=options)

    if options.compile_depth == CompileDepth.COMPILE_STABLEHLO_OP_BY_OP:
        # lower to stablehlo and run op-by-op
        if isinstance(gm_or_shlo, str):
            options.compile_depth = CompileDepth.COMPILE_OP_BY_OP
            return _shlo_backend(
                shlo=gm_or_shlo, example_inputs=example_inputs, compiler_config=options
            )
        elif isinstance(gm_or_shlo, torch.fx.GraphModule):
            module, __, gm, graph_constants = torch_to_shlo(
                gm_or_shlo, example_inputs, compiler_config=options
            )
            return _shlo_backend(
                shlo=module,
                example_inputs=example_inputs,
                compiler_config=options,
                gm=gm,
                graph_constants=graph_constants,
            )
        else:
            print("Compiler input not valid", file=sys.stderr)
            exit(1)
    if isinstance(gm_or_shlo, torch.fx.GraphModule):
        return _torch_backend(gm_or_shlo, example_inputs, compiler_config=options)
    print(
        "Reached invalid compile depth in tt_torch/dynamo/backend.py", file=sys.stderr
    )
    exit(1)