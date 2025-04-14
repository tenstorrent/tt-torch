# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import os
import tt_mlir
import sys
import torch_mlir

# from tt_torch.tools.verify import verify_against_golden

from tt_torch.dynamo.torch_backend import (
    TorchExecutor,
    import_graph,
    import_program,
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
    tt_torch_error_message,
)


def create_verify_golden_callback(compiler_config: CompilerConfig):
    # Closure to capture external state in the callback.
    # using CompilerConfig as a context cache

    def verify_golden_callback(binary, callback_context, op_context):
        # Using these parameters, we should be able to query information
        # about the op described by op_context, and its output. I.e. location:
        raw_location = tt_mlir.get_op_loc_info(op_context)

        location = ""
        fused_locations = []

        # Characters to remove
        # handle fused locations
        if "fused" in raw_location:
            fused_locations = raw_location.split("fused")[1]
            fused_locations = fused_locations.split(",")
            fused_locations = [
                # strip unnecessary characters
                loc.translate(str.maketrans("", "", "()[]{}\"' "))
                for loc in fused_locations
            ]

        if "loc(unknown)" in raw_location:
            return

        # there may be multiple source locations so fused_locations has len>=2, the actual "name" of the node is at the end
        location = fused_locations[-1]

        intermediate_data = compiler_config.runtime_intermediate_cache.get(
            location, None
        )

        print("Node Location:", location)

        if intermediate_data is not None:
            print(f"Found golden for op @ {intermediate_data.node.name} == {location}.")

            # return a null tensor for decomposed ops with invalid output tensors (eg. deallocate)
            output_intermediate_tensor = tt_mlir.get_op_output_torch_tensor(
                op_context, callback_context
            )
            print("output intermediate tensor", output_intermediate_tensor)

            if output_intermediate_tensor is not None:
                if output_intermediate_tensor.dim == 0:
                    print("Got scalar as output:", output_intermediate_tensor)
                    output_intermediate_tensor = output_intermediate_tensor.unsqueeze(0)

                intermediate_data.decomposed_intermediate_outputs.append(
                    output_intermediate_tensor
                )

            # intermediate_data.decomposed_intermediate_outputs.append(
            #    intermediate_data.golden
            # )  # fake output

            # if intermediate_data.golden != None:
            # print(
            #     f"Decomposition added fake tensor too. Total ct {len(intermediate_data.decomposed_intermediate_outputs)}"
            # )

            # pdb.set_trace()

        # atol = calculate_atol(calculated, golden)
        # if atol > executor.required_atol:
        #     print(f"atol too high for {location}: {atol}")
        # pcc = calculate_pcc(calculated, golden)
        # if pcc < executor.required_pcc:
        #     print(f"pcc too low for {location}: {pcc}")

        # pdb.set_trace()
        # We will need to provide the bindings necesarry in this frontend.
        # Those bindings will interact with the runtime API

    return verify_golden_callback


def dump_module(module, name, compiler_config):
    if compiler_config.dump_info:
        print(f"{name} module", file=sys.stderr)
        print(module, file=sys.stderr)


def _shlo_backend(
    shlo, example_inputs, compiler_config, program=None, graph_constants=None
):
    executor = StablehloExecutor(module=shlo, compiler_config=compiler_config)
    if program is not None:
        # original input is a torch graph
        executor.add_program(program, graph_constants)
    return executor


def _torch_backend(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    with torch.no_grad():
        program, graph_constants = pass_pipeline(gm, example_inputs, compiler_config)
    executor = TorchExecutor(
        program=program,
        graph_constants=graph_constants,
        compiler_config=compiler_config,
    )
    return executor


def torch_to_shlo(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    with torch.no_grad():
        program, graph_constants = pass_pipeline(gm, example_inputs, compiler_config)

    module = import_program(program)
    verify_ir(module)

    dump_module(module=module, name="Torch FX module", compiler_config=compiler_config)

    if compiler_config.profile_ops:
        compiler_config.set_torch_mlir_module(module.operation.get_asm())

    run_pipeline_with_repro_report(
        module,
        f"builtin.module(torchdynamo-export-to-torch-backend-pipeline)",
        "Lowering TorchFX IR -> Torch Backend IR",
        compiler_config.dump_debug,
    )
    dump_module(
        module=module, name="Torch Backend module", compiler_config=compiler_config
    )

    lower_mlir_module(False, OutputType.STABLEHLO, module)

    dump_module(module=module, name="StableHLO module", compiler_config=compiler_config)

    return module, program, graph_constants


def shlo_to_flatbuffer(executor, module, compiler_config, example_inputs):

    if compiler_config.profile_ops:
        compiler_config.set_stablehlo_mlir_module(module.operation.get_asm())

    ttir = tt_mlir.compile_stable_hlo_to_ttir(
        module.operation.get_asm(enable_debug_info=True)
    )
    dump_module(module=ttir, name="TTIR module", compiler_config=compiler_config)

    if compiler_config.enable_intermediate_verification:
        executor.register_intermediate_callback(
            create_verify_golden_callback(compiler_config)
        )

    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
    dump_module(module=ttnn, name="TTNN module", compiler_config=compiler_config)

    return binary


def _base_backend(gm, example_inputs, compiler_config):
    shlo, program, graph_constants = torch_to_shlo(gm, example_inputs, compiler_config)
    executor = Executor(program, graph_constants, compiler_config)

    compiler_config.record_property("achieved_compile_depth", "STABLEHLO")

    if compiler_config.compile_depth == CompileDepth.STABLEHLO:
        return executor

    binary = shlo_to_flatbuffer(executor, shlo, compiler_config, example_inputs)
    executor.set_binary(binary)

    compiler_config.record_property("achieved_compile_depth", "TTNN_IR")

    return executor


@tt_torch_error_message
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
            module, program, graph_constants = torch_to_shlo(
                gm, example_inputs, compiler_config=options
            )
            return _shlo_backend(
                shlo=module,
                example_inputs=example_inputs,
                compiler_config=options,
                program=program,
                graph_constants=graph_constants,
            )

    return _base_backend(gm, example_inputs, compiler_config=options)
