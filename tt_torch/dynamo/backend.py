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


def _shlo_backend(shlo, example_inputs, compiler_config, gm=None, graph_constants=None):
    executor = StablehloExecutor(module=shlo, compiler_config=compiler_config)
    if gm is not None:
        # original input is a torch graph
        executor.add_gm(gm, graph_constants)
    return executor


def _torch_backend(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    # Apply environment overrides at start of compilation to allow overriding what was set in the test
    compiler_config.apply_environment_overrides()
    with torch.no_grad():
        gm, graph_constants = pass_pipeline(gm, example_inputs, compiler_config)
    executor = TorchExecutor(
        gm=gm, graph_constants=graph_constants, compiler_config=compiler_config
    )
    return executor


def torch_to_shlo(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    with torch.no_grad():
        gm, graph_constants = pass_pipeline(gm, example_inputs, compiler_config)
    dump_intermediates = os.environ.get("TT_TORCH_IR_LOG_LEVEL")
    dump_info = False
    dump_debug = False
    if dump_intermediates:
        dump_debug = dump_intermediates == "DEBUG"
        dump_info = dump_debug or dump_intermediates == "INFO"

    module = import_graph(gm.graph)
    verify_ir(module)

    if dump_info:
        print("Torch module", file=sys.stderr)
        module.dump()

    if compiler_config.profile_ops:
        compiler_config.set_torch_mlir_module(module.operation.get_asm())

    lower_to_stable_hlo(module, enable_ir_printing=dump_debug)
    if dump_info:
        print("StableHLO module", file=sys.stderr)
        module.dump()

    return module, gm, graph_constants


def shlo_to_flatbuffer(executor, module, compiler_config):
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
    compiler_config.apply_environment_overrides()
    if isinstance(gm_or_shlo, torch.fx.GraphModule):
        shlo, gm, graph_constants = torch_to_shlo(
            gm_or_shlo, example_inputs, compiler_config
        )
    # input is a stablehlo string module
    elif isinstance(gm_or_shlo, str):
        shlo = parse_module_from_str(gm_or_shlo)
        gm = None
        graph_constants = None
    else:
        assert False, "Compiler input not valid"

    executor = Executor(gm, graph_constants, compiler_config)

    if compiler_config.compile_depth == CompileDepth.STABLEHLO:
        return executor

    binary = shlo_to_flatbuffer(executor, shlo, compiler_config)
    executor.set_binary(binary)
    return executor


def backend(gm_or_shlo, example_inputs, options=None):
    if options is None:
        options = CompilerConfig()
    if (
        options.compile_depth == CompileDepth.COMPILE_OP_BY_OP
        or options.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
    ):
        if options.op_by_op_backend == OpByOpBackend.TORCH:
            assert isinstance(gm_or_shlo, torch.fx.GraphModule)
            return _torch_backend(gm_or_shlo, example_inputs, compiler_config=options)
        else:
            gm = None
            graph_constants = None
            module = gm_or_shlo
            if isinstance(gm_or_shlo, torch.fx.GraphModule):
                module, gm, graph_constants = torch_to_shlo(
                    gm_or_shlo, example_inputs, compiler_config=options
                )
            return _shlo_backend(
                shlo=module,
                example_inputs=example_inputs,
                compiler_config=options,
                gm=gm,
                graph_constants=graph_constants,
            )


    return _base_backend(gm_or_shlo, example_inputs, compiler_config=options)
