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
    verify_ir,
    lower_to_stable_hlo,
)
from tt_torch.dynamo.shlo_backend import (
    generate_random_inputs_for_shlo,
    parse_module_from_str,
    StablehloExecutor,
)

from torch_mlir.ir import Module

from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
    lower_mlir_module,
)
from tt_torch.dynamo.passes import pass_pipeline
from tt_torch.dynamo.executor import Executor
from tt_torch.tools.utils import (
    OpByOpBackend,
    CompilerConfig,
    CompileDepth,
)


def verify_golden_callback(binary, callback_context, op_context):
    # Using these parameters, we should be able to query information
    # about the op described by op_context, and its output. I.e. location:
    location = tt_mlir.get_op_loc_info(op_context)
    # ...

    # We will need to provide the bindings necesarry in this frontend.
    # Those bindings will interact with the runtime API


def dump_module(module, name, compiler_config):
    if compiler_config.dump_info:
        print(f"{name} module", file=sys.stderr)
        if isinstance(module, torch_mlir.ir.Operation):
            module.print(large_elements_limit=0)
        elif isinstance(module, str):
            print(module)
        else:
            raise TypeError(f"Unsupported module type: {type(module)}")


def _shlo_backend(shlo, example_inputs, compiler_config, gm=None, graph_constants=None):
    executor = StablehloExecutor(module=shlo, compiler_config=compiler_config)
    if gm is not None:
        # original input is a torch graph
        executor.add_gm(gm, graph_constants)
    return executor


def _torch_backend(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    with torch.no_grad():
        gm, graph_constants = pass_pipeline(gm, example_inputs, compiler_config)
    executor = TorchExecutor(
        gm=gm, graph_constants=graph_constants, compiler_config=compiler_config
    )
    return executor


def generate_torch_fx_ir(
    gm: torch.fx.GraphModule, example_inputs, compiler_config: CompilerConfig
):
    with torch.no_grad():
        gm, graph_constants = pass_pipeline(gm, example_inputs, compiler_config)

    module = import_graph(gm.graph)
    verify_ir(module)

    dump_module(module=module, name="Torch FX module", compiler_config=compiler_config)

    if compiler_config.profile_ops:
        compiler_config.set_torch_mlir_module(module.operation.get_asm())

    return module, gm, graph_constants


def torch_fx_ir_to_torch_backend_ir(module: Module, compiler_config: CompilerConfig):
    run_pipeline_with_repro_report(
        module,
        f"builtin.module(torchdynamo-export-to-torch-backend-pipeline)",
        "Lowering TorchFX IR -> Torch Backend IR",
        compiler_config.dump_debug,
    )
    dump_module(
        module=module, name="Torch Backend module", compiler_config=compiler_config
    )
    return module


def torch_backend_ir_to_stablehlo(module: Module, compiler_config: CompilerConfig):
    lower_mlir_module(False, OutputType.STABLEHLO, module)
    dump_module(module=module, name="StableHLO module", compiler_config=compiler_config)
    return module


def stablehlo_to_ttir(module: Module, compiler_config: CompilerConfig):
    ttir = tt_mlir.compile_stable_hlo_to_ttir(
        module.operation.get_asm(enable_debug_info=True)
    )
    dump_module(module=ttir, name="TTIR module", compiler_config=compiler_config)
    return ttir


def ttir_to_ttnn_and_binary(ttir, compiler_config: CompilerConfig):
    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
    dump_module(module=ttnn, name="TTNN module", compiler_config=compiler_config)
    return binary, ttnn


def _base_backend(gm, example_inputs, compiler_config):
    module, gm, graph_constants = generate_torch_fx_ir(
        gm, example_inputs, compiler_config
    )
    executor = Executor(gm, graph_constants, compiler_config)
    if compiler_config.compile_depth == CompileDepth.TORCH_FX:
        return executor

    module = torch_fx_ir_to_torch_backend_ir(module, compiler_config)
    if compiler_config.profile_ops:
        compiler_config.set_torch_mlir_module(module.operation.get_asm())
    if compiler_config.compile_depth == CompileDepth.TORCH_BACKEND_IR:
        return executor

    module = torch_backend_ir_to_stablehlo(module, compiler_config)
    if compiler_config.profile_ops:
        compiler_config.set_stablehlo_mlir_module(module.operation.get_asm())
    if compiler_config.compile_depth == CompileDepth.STABLEHLO:
        return executor

    module = stablehlo_to_ttir(module, compiler_config)
    if compiler_config.compile_depth == CompileDepth.TTIR_DIALECT:
        return executor

    binary, ttnn = ttir_to_ttnn_and_binary(module, compiler_config)
    if compiler_config.compile_depth == CompileDepth.TTNN_DIALECT:
        return executor

    executor.set_binary(binary)
    return executor


def backend(gm, example_inputs, options=None):
    assert isinstance(gm, torch.fx.GraphModule), "Backend only supports torch graphs"

    if options is None:
        options = CompilerConfig()

    # Apply environment overrides at start of compilation to allow overriding what was set in the test
    options.apply_environment_overrides()

    if (
        options.compile_depth == CompileDepth.COMPILE_OP_BY_OP
        or options.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
    ):
        if options.op_by_op_backend == OpByOpBackend.TORCH:
            # run torch graph op-by-op
            return _torch_backend(gm, example_inputs, compiler_config=options)
        else:
            # op_by_op_backend == OpByOpBackend.STABLEHLO
            # convert torch to stablehlo, then run stablehlo op-by-op
            module, gm, graph_constants = generate_torch_fx_ir(
                gm, example_inputs, compiler_config=options
            )
            module = torch_fx_ir_to_torch_backend_ir(module, compiler_config=options)
            module = torch_backend_ir_to_stablehlo(module, compiler_config=options)
            return _shlo_backend(
                shlo=module,
                example_inputs=example_inputs,
                compiler_config=options,
                gm=gm,
                graph_constants=graph_constants,
            )
    return _base_backend(gm, example_inputs, compiler_config=options)
