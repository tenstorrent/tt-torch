# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import time
from tt_torch.tools.utils import CompilerConfig
from tt_torch.tools.verify import calculate_pcc
from tt_torch.tools.device_manager import DeviceManager
from tt_torch.dynamo.backend import backend, BackendOptions
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StaticCache,
)
from tests.utils import clear_dynamo_cache
<<<<<<< HEAD
=======
from transformers.models.llama.modeling_llama import LlamaModel
import os
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
>>>>>>> 83461321 (setup xla env stuff from het)
import tt_mlir
import numpy as np
_global_max_cache_len = 64 + 64


def load_model(model_name="meta-llama/Llama-3.2-3B"):
    # set up the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=True,
    )
<<<<<<< HEAD
=======
    model.config.num_hidden_layers=2
>>>>>>> 83461321 (setup xla env stuff from het)

    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer


def load_inputs(
    model,
    tokenizer,
    # test_input="This is a sample text from ",
    test_input="I like taking walks in the",
    max_cache_len=_global_max_cache_len,
):
    batch_size = 1
    inputs = tokenizer.encode_plus(
        test_input,
        return_tensors="pt",
        truncation=True,
    )

    # set up static cache
    static_cache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=model.device,
        dtype=model.dtype,
    )

    cache_position = torch.arange(0, inputs.input_ids.shape[1])

    args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "use_cache": True,
        "cache_position": cache_position,
    }
    return args


def setup_xla_environment():
    """Setup XLA environment for tensor parallelism."""
    print("Setting up XLA environment...")
    num_devices = xr.global_runtime_device_count()

    # Basic XLA configuration
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE" # Enables the auto parallel pass in tt-mlir
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1" # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ["MESH_SHAPE"] = f"1,{num_devices}" # Sets the mesh shape used by the auto parallel pass

    # Initialize SPMD
    xr.use_spmd()
    print("XLA environment configured.")

def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.

    Args:
        num_devices: Total number of devices
        mesh_shape: Shape of the device mesh (batch_dim, model_dim)

    Returns:
        Mesh object for SPMD operations
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh

@torch.inference_mode()
def test_llama3_generate():
    # Initialize model and inputs
    start_time = time.time()
    model, tokenizer = load_model()
    input_args = load_inputs(model, tokenizer)
    golden_input_args = input_args.copy()
    generated_ids = input_args["input_ids"]
    model_load_time = time.time() - start_time

    initial_prompt = tokenizer.decode(generated_ids[0].tolist())
    print(f"Initial prompt: '{initial_prompt}'")

    # setup XLA environment and device mesh
    setup_xla_environment()
    mesh = create_device_mesh()
    
    # apply shardings
    
    
    
    # Allow local disablement of golden verification to accelerate tests
    # by avoiding dev2host transfer of static cache    
    enable_golden = False

    # Setup compilation
    clear_dynamo_cache()
    cc = CompilerConfig()

    # Consteval disabled due to 4D Causal Attention Mask evaluation getting constant folded in torchfx
    #   due to incorrect tracing of static cache and malformed output missing static cache tensors
    cc.enable_consteval = True
    cc.consteval_parameters = True

    options = BackendOptions()
    options.compiler_config = cc

    # device = DeviceManager.create_parent_mesh_device(mesh_shape=[1, 1])
    # options.devices = [device]

    buffer_cache = {}
    options.buffer_cache = buffer_cache

    constant_cache = {}
    options.constant_cache = constant_cache

    # _backend = backend 
    _backend = 'tt-experimental' 
    

    compiled_model = torch.compile(
        model, backend=_backend, dynamic=False, options=options
    )

    # Token generation with data collection
    tokens_to_generate = 3
    golden_pccs = []
    cache_pccs_per_iteration = []  # Store cache PCCs for each iteration
    golden_ids = input_args["input_ids"]
    timings = []
    generated_tokens = []
    golden_generated_tokens = []  # Track golden tokens separately

    print(initial_prompt, end="", flush=True)

    generation_start = time.time()

    for i in range(tokens_to_generate):
        iteration_start = time.time()

        # Execute model
        outputs = compiled_model(**input_args)

        # Update inputs for next iteration
        cache_position = input_args["cache_position"][-1:] + 1

        # Golden calculation - Adds execution time to transfer static caches to host.
        if enable_golden:
            golden_outputs = model(**golden_input_args)
            next_golden_ids = golden_outputs.logits[:, -1:].argmax(dim=-1)

            # Collect golden token
            golden_token = tokenizer.decode(next_golden_ids[0].tolist())
            golden_generated_tokens.append(golden_token)

            # Calculate golden PCCs (only if enabled)
            golden_pcc = calculate_pcc(golden_outputs.logits, outputs.logits)
            golden_pccs.append(golden_pcc)

            # Calculate golden PCCs from static cache internals
            flat_static_cache = []
            static_cache_pccs = []
            for kcache, vcache in zip(
                golden_outputs.past_key_values.key_cache,
                golden_outputs.past_key_values.value_cache,
            ):
                flat_static_cache.extend(kcache)
                flat_static_cache.extend(vcache)

            for torch_to_runtime_tensors in buffer_cache.values():
                for j, runtime_buffer in enumerate(torch_to_runtime_tensors.values()):
                    runtime_static_cache = tt_mlir.to_host(
                        runtime_buffer, deallocate_tensor=False
                    )[0]

                    # Calculate PCC between golden and runtime static cache
                    static_cache_pcc = calculate_pcc(
                        flat_static_cache[j], runtime_static_cache
                    )
                    static_cache_pccs.append(static_cache_pcc)

            # Store cache PCCs for this iteration
            cache_pccs_per_iteration.append(static_cache_pccs.copy())

            golden_input_args = {
                "input_ids": next_golden_ids,
                "past_key_values": golden_outputs.past_key_values,
                "use_cache": True,
                "cache_position": cache_position,
            }
        else:
            # Add empty entries when golden verification is disabled
            golden_pccs.append(0.0)
            cache_pccs_per_iteration.append([])

        # Post-processing
        next_token_ids = outputs.logits[:, -1:].argmax(dim=-1)
        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

        # Decode and collect token
        new_token = tokenizer.decode(next_token_ids[0].tolist())
        generated_tokens.append(new_token)
        print(new_token, end="", flush=True)

        input_args = {
            "input_ids": next_token_ids,
            "past_key_values": input_args["past_key_values"],  # updated in place
            "cache_position": cache_position,
            "use_cache": True,
        }

        total_iteration_time = time.time() - iteration_start

        # Store timing information
        phase = "prefill" if i == 0 else "decode"
        timings.append(
            {
                "iteration": i,
                "phase": phase,
                "total_time": total_iteration_time,
                "token": new_token,
            }
        )

    total_generation_time = time.time() - generation_start
    generated_text = initial_prompt + "".join(generated_tokens)
    golden_generated_text = (
        initial_prompt + "".join(golden_generated_tokens) if enable_golden else ""
    )

    # Display summary at the end
    display_summary(
        initial_prompt=initial_prompt,
        model_load_time=model_load_time,
        total_generation_time=total_generation_time,
        tokens_to_generate=tokens_to_generate,
        timings=timings,
        golden_pccs=golden_pccs,
        cache_pccs_per_iteration=cache_pccs_per_iteration,
        generated_text=generated_text,
        golden_generated_text=golden_generated_text,
        enable_golden=enable_golden,
    )

    # Cleanup
    # DeviceManager.release_parent_device(device)
    clear_dynamo_cache()


def display_summary(
    initial_prompt,
    model_load_time,
    total_generation_time,
    tokens_to_generate,
    timings,
    golden_pccs,
    cache_pccs_per_iteration,
    generated_text,
    golden_generated_text,
    enable_golden=True,
):
    """Display summary of the generation loop timing and accuracy."""
    print()  # Add a newline at the end of the output
    print("=" * 80)
    print("GENERATION SUMMARY")
    print("=" * 80)

    # Calculate statistics
    prefill_times = [t["total_time"] for t in timings if t["phase"] == "prefill"]
    decode_times = [t["total_time"] for t in timings if t["phase"] == "decode"]

    print(f"Initial prompt: '{initial_prompt}'")
    print(f"Generated text: '{generated_text}'")
    if enable_golden:
        print(f"Golden text: '{golden_generated_text}'")
    print()

    print(f"Model loading time: {model_load_time:.3f}s")
    print(f"Total generation time: {total_generation_time:.3f}s")
    print(f"Tokens generated: {tokens_to_generate}")
    print(f"Average tokens/second: {tokens_to_generate / total_generation_time:.2f}")
    print()

    if prefill_times:
        print(f"PREFILL (first token):")
        print(f"  Time: {prefill_times[0]:.3f}s")
        print()

    if decode_times:
        print(f"DECODE (subsequent tokens):")
        print(f"  Count: {len(decode_times)}")
        print(f"  Average time: {sum(decode_times) / len(decode_times):.3f}s")
        print(f"  Min time: {min(decode_times):.3f}s")
        print(f"  Max time: {max(decode_times):.3f}s")
        print(f"  Average tokens/second: {len(decode_times) / sum(decode_times):.2f}")
        print()

    if enable_golden and golden_pccs and any(pcc > 0 for pcc in golden_pccs):
        print(f"ACCURACY:")
        average_pcc = sum(golden_pccs) / len(golden_pccs)

        assert (
            average_pcc >= 0.6
        ), "Average PCC for all logit vectors at all decode steps generated for this prompt should be at least 0.6."

        print(f"  Average PCC: {average_pcc:.6f}")
        print(f"  Min PCC: {min(golden_pccs):.6f}")
        print(f"  Max PCC: {max(golden_pccs):.6f}")
        print()

    # Show timing progression for all iterations with cache PCCs
    print("DETAILED TIMING (all iterations):")
    if enable_golden:
        print(
            "Iter | Phase   | Total   | PCC      | Cache PCCs                    | Token"
        )
    else:
        print("Iter | Phase   | Total   | Token")
    print("-" * (80 if enable_golden else 50))

    def color_pcc(pcc):
        if pcc < 0.9:
            return f"\033[91m{pcc:.3f}\033[0m"  # Red
        elif pcc < 0.95:
            return f"\033[93m{pcc:.3f}\033[0m"  # Yellow
        elif pcc < 0.99:
            return f"\033[93m{pcc:.3f}\033[0m"  # Yellow
        else:
            return f"\033[92m{pcc:.3f}\033[0m"  # Green

    for i, timing in enumerate(timings):
        token_repr = (
            repr(timing["token"])[:8] + "..."
            if len(repr(timing["token"])) > 11
            else repr(timing["token"])
        )

        if enable_golden:
            pcc = golden_pccs[i] if i < len(golden_pccs) else 0.0

            # Format cache PCCs with colors
            cache_pccs_str = ""
            if i < len(cache_pccs_per_iteration) and cache_pccs_per_iteration[i]:
                colored_cache_pccs = [
                    color_pcc(pcc) for pcc in cache_pccs_per_iteration[i]
                ]
                cache_pccs_str = "[" + ",".join(colored_cache_pccs) + "]"
            else:
                cache_pccs_str = "[]"

            print(
                f"{i:4d} | {timing['phase']:7s} | {timing['total_time']:6.3f}s | {pcc:.6f} | {cache_pccs_str:29s} | {token_repr}"
            )
        else:
            print(
                f"{i:4d} | {timing['phase']:7s} | {timing['total_time']:6.3f}s | {token_repr}"
            )
