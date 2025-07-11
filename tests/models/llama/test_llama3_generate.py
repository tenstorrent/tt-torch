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

_global_max_cache_len = 64 + 64


def load_model(model_name="meta-llama/Llama-3.2-3B"):
    # set up the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer


def load_inputs(
    model,
    tokenizer,
    test_input="This is a sample text from ",
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


def display_summary(
    initial_prompt,
    model_load_time,
    total_generation_time,
    tokens_to_generate,
    timings,
    golden_pccs,
    generated_text,
):
    """Display comprehensive summary of the generation test."""
    print()  # Add a newline at the end of the output
    print("=" * 80)
    print("GENERATION SUMMARY")
    print("=" * 80)

    # Calculate statistics
    prefill_times = [t["total_time"] for t in timings if t["phase"] == "prefill"]
    decode_times = [t["total_time"] for t in timings if t["phase"] == "decode"]
    inference_times = [t["inference_time"] for t in timings]

    print(f"Initial prompt: '{initial_prompt}'")
    print(f"Generated text: '{generated_text}'")
    print()

    print(f"Model loading time: {model_load_time:.3f}s")
    print(f"Total generation time: {total_generation_time:.3f}s")
    print(f"Tokens generated: {tokens_to_generate}")
    print(f"Average tokens/second: {tokens_to_generate / total_generation_time:.2f}")
    print()

    if prefill_times:
        print(f"PREFILL (first token):")
        print(f"  Time: {prefill_times[0]:.3f}s")
        print(f"  Inference: {timings[0]['inference_time']:.3f}s")
        print(f"  Post-process: {timings[0]['post_process_time']:.3f}s")
        print()

    if decode_times:
        print(f"DECODE (subsequent tokens):")
        print(f"  Count: {len(decode_times)}")
        print(f"  Average time: {sum(decode_times) / len(decode_times):.3f}s")
        print(f"  Min time: {min(decode_times):.3f}s")
        print(f"  Max time: {max(decode_times):.3f}s")
        print(f"  Average tokens/second: {len(decode_times) / sum(decode_times):.2f}")
        print()

    print(f"INFERENCE TIMING:")
    print(
        f"  Average inference time: {sum(inference_times) / len(inference_times):.3f}s"
    )
    print(f"  Min inference time: {min(inference_times):.3f}s")
    print(f"  Max inference time: {max(inference_times):.3f}s")
    print()

    print(f"ACCURACY:")
    print(f"  Average PCC: {sum(golden_pccs) / len(golden_pccs):.6f}")
    print(f"  Min PCC: {min(golden_pccs):.6f}")
    print(f"  Max PCC: {max(golden_pccs):.6f}")
    print()

    # Show timing progression for first 10 iterations
    print("DETAILED TIMING (first 10 iterations):")
    print("Iter | Phase   | Inference | Post-Proc | Total   | PCC      | Token")
    print("-" * 70)
    for i, timing in enumerate(timings[:10]):
        token_repr = (
            repr(timing["token"])[:8] + "..."
            if len(repr(timing["token"])) > 11
            else repr(timing["token"])
        )
        pcc = golden_pccs[i] if i < len(golden_pccs) else 0.0
        print(
            f"{i:4d} | {timing['phase']:7s} | {timing['inference_time']:8.3f}s | {timing['post_process_time']:8.3f}s | {timing['total_time']:6.3f}s | {pcc:.6f} | {token_repr}"
        )

    if len(timings) > 10:
        print("...")

    print("=" * 80)


@torch.inference_mode()
def test_llama3_generate():
    print("=" * 80)
    print("Starting Llama 3.2 Generation Test")
    print("=" * 80)

    # Initialize model and inputs
    start_time = time.time()
    model, tokenizer = load_model()
    input_args = load_inputs(model, tokenizer)
    generated_ids = input_args["input_ids"]
    model_load_time = time.time() - start_time

    initial_prompt = tokenizer.decode(generated_ids[0].tolist())
    print(f"Initial prompt: '{initial_prompt}'")
    print(f"Model loading time: {model_load_time:.3f}s")
    print("-" * 80)

    # Setup compilation
    compilation_start = time.time()
    clear_dynamo_cache()
    cc = CompilerConfig()
    cc.enable_consteval = False
    cc.consteval_parameters = False

    options = BackendOptions()
    options.compiler_config = cc

    device = DeviceManager.create_parent_mesh_device(mesh_shape=[1, 1])
    options.devices = [device]

    buffer_cache = {}
    options.buffer_cache = buffer_cache

    constant_cache = {}
    options.constant_cache = constant_cache

    compiled_model = torch.compile(
        model, backend=backend, dynamic=False, options=options
    )

    # Token generation with data collection
    tokens_to_generate = 32
    golden_pccs = []
    golden_ids = input_args["input_ids"]
    timings = []
    generated_tokens = []

    print(initial_prompt, end="", flush=True)

    generation_start = time.time()

    for i in range(tokens_to_generate):
        iteration_start = time.time()

        # Golden calculation
        golden_outputs = model(**input_args)
        next_golden_ids = golden_outputs.logits[:, -1:].argmax(dim=-1)

        # Execute model
        outputs = compiled_model(**input_args)
        inference_time = time.time() - iteration_start

        golden_pcc = calculate_pcc(golden_outputs.logits, outputs.logits)
        golden_pccs.append(golden_pcc)

        # Post-processing
        post_process_start = time.time()
        next_token_ids = outputs.logits[:, -1:].argmax(dim=-1)
        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

        # Decode and collect token
        new_token = tokenizer.decode(next_token_ids[0].tolist())
        generated_tokens.append(new_token)
        print(new_token, end="", flush=True)

        # Update inputs for next iteration
        cache_position = input_args["cache_position"][-1:] + 1
        input_args = {
            "input_ids": next_token_ids.to(dtype=torch.int32),
            "past_key_values": input_args["past_key_values"],  # updated in place
            "cache_position": cache_position,
            "use_cache": True,
        }

        post_process_time = time.time() - post_process_start
        total_iteration_time = time.time() - iteration_start

        # Store timing information
        phase = "prefill" if i == 0 else "decode"
        timings.append(
            {
                "iteration": i,
                "phase": phase,
                "inference_time": inference_time,
                "post_process_time": post_process_time,
                "total_time": total_iteration_time,
                "token": new_token,
            }
        )

    total_generation_time = time.time() - generation_start
    generated_text = initial_prompt + "".join(generated_tokens)

    # Display summary at the end
    display_summary(
        initial_prompt=initial_prompt,
        model_load_time=model_load_time,
        total_generation_time=total_generation_time,
        tokens_to_generate=tokens_to_generate,
        timings=timings,
        golden_pccs=golden_pccs,
        generated_text=generated_text,
    )

    # Cleanup
    DeviceManager.release_parent_device(device)
    clear_dynamo_cache()

    print("Test completed successfully!")
