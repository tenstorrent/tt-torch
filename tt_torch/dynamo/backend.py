# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import os
import warnings
import tt_mlir
import sys
import torch_mlir

from tt_torch.tools.utils import (
    OpByOpBackend,
    CompilerConfig,
    CompileDepth,
    tt_torch_error_message,
    sanitize_filename,
)


class BackendOptions:
    def __init__(
        self,
        compiler_config=CompilerConfig(),
        devices=[None],
        async_mode=False,
        buffer_cache=None,
        constant_cache=None,
    ):
        self.compiler_config = compiler_config
        self.devices = devices
        self.async_mode = async_mode

        # These caches store runtime tensors reprsenting buffers and graph constants
        # and are reused between executors, as long as they share the same BackendOptions object

        self.buffer_cache = buffer_cache if buffer_cache is not None else {}
        self.constant_cache = constant_cache if constant_cache is not None else {}

    def clear_caches(self):
        # Deallocate runtime tensors when the BackendOptions object is deleted\
        if self.constant_cache is not None:
            for device_weights in self.constant_cache.keys():
                for runtime_weight in device_weights.keys():
                    tt_mlir.deallocate_tensor(runtime_weight, force=True)

        if self.buffer_cache is not None:
            for device_buffers in self.buffer_cache.keys():
                for runtime_buffer in device_buffers.keys():
                    tt_mlir.deallocate_tensor(runtime_buffer, force=True)

    def __del__(self):
        self.clear_caches()


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


def create_verify_golden_callback(compiler_config: CompilerConfig):
    # Closure to capture external state in the callback.
    # using CompilerConfig as a context cache

    def verify_golden_callback(binary, callback_context, op_context):

        raw_location = tt_mlir.get_op_loc_info(op_context)

        location = ""
        fused_locations = []

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

        # there may be multiple source locations so fused_locations, the actual "name" of the node is at the end
        location = fused_locations[-1]

        intermediate_data = compiler_config.runtime_intermediate_cache.get(
            location, None
        )

        if intermediate_data is not None:
            print(f"Found golden for op @ {intermediate_data.node.name} == {location}.")

            # return a null tensor for decomposed ops with invalid output tensors (eg. deallocate)
            output_intermediate_tensor = tt_mlir.get_op_output_torch_tensor(
                op_context, callback_context
            )

            if output_intermediate_tensor is not None:
                if output_intermediate_tensor.dim == 0:
                    output_intermediate_tensor = output_intermediate_tensor.unsqueeze(0)

                intermediate_data.decomposed_intermediate_outputs.append(
                    output_intermediate_tensor
                )

    return verify_golden_callback


def dump_module(module, name, compiler_config):
    if compiler_config.dump_info:
        print(f"{name} module", file=sys.stderr)
        print(module, file=sys.stderr)

    if compiler_config.save_mlir_override and name.lower() in (
        n.lower() for n in compiler_config.save_mlir_override
    ):
        if not compiler_config.model_name:
            print("Cannot dump module, no model name provided for save_mlir_override")
            return
        assert (
            compiler_config.output_mlir_dir
        ), "Cannot dump module, no output directory provided"
        output_dir = compiler_config.output_mlir_dir
        sanitized_model_name = sanitize_filename(compiler_config.model_name)
        filepath = os.path.join(
            output_dir, f"{sanitized_model_name}_{name.lower()}.mlir"
        )
        with open(filepath, "a") as f:
            if isinstance(module, str):
                f.write(module)
            else:
                f.write(module.operation.get_asm())
            f.write("\n")


def _shlo_backend(
    mcg,
    example_inputs,
    compiler_config,
    devices=None,
    async_mode=False,
):
    executor = StablehloExecutor(
        module=mcg.shlo_modules[0],
        compiler_config=compiler_config,
        devices=devices,
        async_mode=async_mode,
    )
    executor.add_program(mcg)
    return executor


def _torch_backend(
    gm: torch.fx.GraphModule, example_inputs, compiler_config, devices, async_mode
):
    with torch.no_grad():
        mcg = pass_pipeline(gm, example_inputs, compiler_config)

    executor = TorchExecutor(
        mcg=mcg,
        compiler_config=compiler_config,
        devices=devices,
        async_mode=async_mode,
    )
    return executor


def torch_to_shlo(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    with torch.no_grad():
        mcg = pass_pipeline(gm, example_inputs, compiler_config)

    for device_idx, program in mcg.programs.items():
        module = import_program(program)
        verify_ir(module)

        dump_module(module=module, name="Torch FX", compiler_config=compiler_config)

        if compiler_config.profile_ops:
            compiler_config.set_torch_mlir_module(module.operation.get_asm())

        run_pipeline_with_repro_report(
            module,
            f"builtin.module(torchdynamo-export-to-torch-backend-pipeline)",
            "Lowering TorchFX IR -> Torch Backend IR",
            compiler_config.dump_debug,
        )
        dump_module(
            module=module, name="Torch Backend", compiler_config=compiler_config
        )

        lower_mlir_module(False, OutputType.STABLEHLO, module)

        dump_module(module=module, name="StableHLO", compiler_config=compiler_config)

        mcg.shlo_modules[device_idx] = module
    return mcg


def shlo_to_flatbuffer(
    executor,
    system_desc_path,
    module,
    compiler_config,
    len_activations,
    len_graph_constants,
):

    if compiler_config.profile_ops:
        compiler_config.set_stablehlo_mlir_module(module.operation.get_asm())

    shlo = module.operation.get_asm(enable_debug_info=True)
    if compiler_config.automatic_parallelization:
        shlo = tt_mlir.stable_hlo_automatic_parallelization(
            shlo, compiler_config.mesh_shape, len_activations, len_graph_constants
        )
        dump_module(
            module=shlo,
            name="STABLEHLO_AUTOMATIC_PARALLELIZATION",
            compiler_config=compiler_config,
        )

    ttir = tt_mlir.compile_stable_hlo_to_ttir(shlo)
    dump_module(module=ttir, name="TTIR", compiler_config=compiler_config)

    if compiler_config.enable_intermediate_verification:
        executor.register_intermediate_callback(
            create_verify_golden_callback(compiler_config)
        )

    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(
        ttir,
        system_desc_path,
        len_activations,
        len_graph_constants,
        compiler_config.enable_consteval,
    )
    dump_module(module=ttnn, name="TTNN", compiler_config=compiler_config)

    return binary


def _base_backend(
    gm,
    example_inputs,
    compiler_config,
    devices,
    async_mode,
    buffer_cache,
    constant_cache,
):
    mcg = torch_to_shlo(gm, example_inputs, compiler_config)
    executor = Executor(
        mcg,
        compiler_config,
        devices=devices,
        async_mode=async_mode,
        buffer_cache=buffer_cache,
        constant_cache=constant_cache,
    )

    compiler_config.record_property("achieved_compile_depth", "STABLEHLO")

    if compiler_config.compile_depth == CompileDepth.STABLEHLO:
        return executor

    for i, shlo in mcg.shlo_modules.items():
        binary_bytestream = shlo_to_flatbuffer(
            executor,
            executor.system_desc_paths[i],
            shlo,
            compiler_config,
            len(mcg.example_inputs[i]) + len(mcg.buffers[i]),
            len(mcg.constant_inputs[i]),
        )
        mcg.binaries[i] = tt_mlir.create_binary_from_bytestream(binary_bytestream)

    compiler_config.record_property("achieved_compile_depth", "TTNN_IR")
    return executor


@tt_torch_error_message
def backend(gm, example_inputs, options: BackendOptions = None):
    warnings.filterwarnings("ignore", message="Failed to fetch module*")
    assert isinstance(gm, torch.fx.GraphModule), "Backend only supports torch graphs"

    if options is None:
        cc = CompilerConfig()
        devices = None
        async_mode = False

        # If the backend is called without options,
        # tt-torch runtime tensor caches are default-disabled.
        buffer_cache = None
        constant_cache = None
    else:
        cc = options.compiler_config
        devices = options.devices
        async_mode = options.async_mode
        buffer_cache = options.buffer_cache
        constant_cache = options.constant_cache

    # Apply environment overrides at start of compilation to allow overriding what was set in the test
    cc.apply_environment_overrides()

    if (
        cc.compile_depth == CompileDepth.COMPILE_OP_BY_OP
        or cc.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
    ):
        if cc.op_by_op_backend == OpByOpBackend.TORCH:
            # run torch graph op-by-op
            return _torch_backend(
                gm,
                example_inputs,
                compiler_config=cc,
                devices=devices,
                async_mode=async_mode,
            )
        else:
            # op_by_op_backend == OpByOpBackend.STABLEHLO
            # convert torch to stablehlo, then run stablehlo op-by-op
            mcg = torch_to_shlo(gm, example_inputs, compiler_config=cc)
            return _shlo_backend(
                mcg=mcg,
                example_inputs=example_inputs,
                compiler_config=cc,
                devices=devices,
                async_mode=async_mode,
            )
    return _base_backend(
        gm,
        example_inputs,
        compiler_config=cc,
        devices=devices,
        async_mode=async_mode,
        buffer_cache=buffer_cache,
        constant_cache=constant_cache,
    )
