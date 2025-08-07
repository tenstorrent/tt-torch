# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# We must import tt_torch here as its import will register tenstorrents PJRT plugin with torch-xla.
import tt_torch

import os
import sys
import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
import torch_xla.core.xla_model as xm
from torch_xla.distributed.spmd import Mesh
import numpy as np
from typing import Tuple, Union
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
PROMPT = "What is the name of the largest planet in our solar system?"

def setup_xla_environment():
    """Setup XLA environment for tensor parallelism."""
    print("Setting up XLA environment...")

    # Basic XLA configuration
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE" # Enables the auto parallel pass in tt-mlir
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1" # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ["MESH_SHAPE"] = "1,8" # Sets the mesh shape used by the auto parallel pass

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


def _apply_tensor_parallel_sharding_to_base_model(model: LlamaModel, mesh: Mesh) -> None:
    for i, layer in enumerate(model.layers):
        print(f"Sharding layer {i+1}/{len(model.layers)}")

        # Column parallel: Split output dimension across devices
        # up_proj: [hidden_size, intermediate_size] -> shard dim 0
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", "batch"))

        # gate_proj: [hidden_size, intermediate_size] -> shard dim 0
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", "batch"))

        # Row parallel: Split input dimension across devices
        # down_proj: [intermediate_size, hidden_size] -> shard dim 1
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, ("batch", "model"))

        # Column parallel: Split attention heads across devices
        # q_proj: [hidden_size, num_heads * head_dim] -> shard dim 0
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", "batch"))

        # k_proj: [hidden_size, num_kv_heads * head_dim] -> shard dim 0
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", "batch"))

        # v_proj: [hidden_size, num_kv_heads * head_dim] -> shard dim 0
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", "batch"))

        # Row parallel: Collect results from all devices
        # o_proj: [num_heads * head_dim, hidden_size] -> shard dim 1
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))

def apply_tensor_parallel_sharding(model: LlamaForCausalLM, mesh: Mesh) -> None:
    model = model.to(torch_xla.device())
    print("LlamaForCausalLM Model: ", model)

    # Shard base Llama model
    print("Applying tensor parallel sharding to LlamaModel...")
    _apply_tensor_parallel_sharding_to_base_model(model.model, mesh)

    # Also shard the language modeling head
    xs.mark_sharding(model.lm_head.weight, mesh, ("model", "batch"))

    print("Tensor parallel sharding applied successfully!")


def generate_single_token(
    model: LlamaForCausalLM,
    inputs: torch.Tensor,
    is_multi_device: bool = False,
    mesh: Union[Mesh, None] = None
):
    if is_multi_device:
        inputs = inputs.to(torch_xla.device())
        xs.mark_sharding(inputs, mesh, (None, None))
    with torch.no_grad():
        outputs = model(input_ids=inputs, output_hidden_states=True)
        if is_multi_device:
            torch_xla.sync(True, True)
        logits = outputs.logits
        if is_multi_device:
            logits = logits.to("cpu")

        # Get next token
        next_token_logits = logits[:, -1, :]
        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
    return next_token_id

def generate_single_token_with_cache(
    model: LlamaForCausalLM,
    inputs: torch.Tensor,
    past_key_values: Union[None, DynamicCache]=None,
    is_multi_device: bool = False,
    mesh: Union[Mesh, None] = None
):
    if is_multi_device:
        inputs = inputs.to(torch_xla.device())
        xs.mark_sharding(inputs, mesh, (None, None))
        
        # If we have past_key_values, move them to XLA device and mark sharding
        if past_key_values is not None:
            # past_key_values = tuple(
            #     tuple(
            #         kv.to(torch_xla.device()) if kv is not None else None
            #         for kv in layer_cache
            #     ) for layer_cache in past_key_values
            # )
            for layer_cache in past_key_values:
                for kv in layer_cache:
                    if kv is not None:
                        kv = kv.to(torch_xla.device())
                    
            # Mark sharding for KV cache tensors
            for layer_idx, (key_cache, value_cache) in enumerate(past_key_values):
                if key_cache is not None and value_cache is not None:
                    # Shard along the head dimension (model parallel)
                    xs.mark_sharding(key_cache, mesh, (None, "model", None, None))
                    xs.mark_sharding(value_cache, mesh, (None, "model", None, None))

    with torch.no_grad():
        # if past_key_values is not None:
        #     torch_xla.sync(True, True)  # Ensure all previous operations are complete
        #     past_key_values = past_key_values.to("cpu")
        #     print("[HET DEBUG] Type of past_key_values: ", type(past_key_values))
        #     print("[HET DEBUG] past_key_values: ", past_key_values)
        #     past_key_values = past_key_values.to(torch_xla.device())
        outputs = model(
            input_ids=inputs, 
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
        
        if is_multi_device:
            torch_xla.sync(True, True)
        
        print("[HET DEBUG] Type of outputs: ", type(outputs))
        print("[HET DEBUG] outputs: ", outputs)
        logits = outputs.logits
        new_past_key_values = outputs.past_key_values
        
        if is_multi_device:
            logits = logits.to("cpu")
            # Move KV cache back to CPU for next iteration
            # new_past_key_values = tuple(
            #     tuple(
            #         kv.to("cpu") if kv is not None else None
            #         for kv in layer_cache
            #     ) for layer_cache in new_past_key_values
            # )
            for layer_cache in new_past_key_values:
                for kv in layer_cache:
                    if kv is not None:
                        kv = kv.to("cpu")

        # Get next token
        next_token_logits = logits[:, -1, :]
        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
    
    return next_token_id, new_past_key_values

def run_text_generation_cpu_multi_token(
    num_tokens: int = 64):
    llama = LlamaForCausalLM.from_pretrained(MODEL_NAME)
    llama = llama.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt", padding=True, truncation=True)
    for step in range(num_tokens):
        print(f"Step {step + 1}/{num_tokens}")

        # Generate next token
        next_token_id = generate_single_token(llama, input_ids)

        if next_token_id == tokenizer.eos_token_id:
            break

        # Append next token to input_ids
        input_ids = torch.cat((input_ids, next_token_id), dim=1)
    
    decoded_text = tokenizer.batch_decode(input_ids)[0]
    
    return decoded_text, input_ids


def run_text_generation_tp_multi_token(
    num_tokens: int = 64
):
    setup_xla_environment()
    mesh = create_device_mesh()
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Applying tensor parallel sharding to CausalLM model...")
    apply_tensor_parallel_sharding(model, mesh)

    input_ids = tokenizer.encode(PROMPT, return_tensors="pt", padding=True, truncation=True)

    for step in range(num_tokens):
        print(f"Step {step + 1}/{num_tokens}")

        # Generate next token
        next_token_id = generate_single_token(model, input_ids, is_multi_device=True, mesh=mesh)
        if next_token_id == tokenizer.eos_token_id:
            break
        # Append next token to input_ids
        input_ids = torch.cat((input_ids, next_token_id), dim=1)

    decoded_text = tokenizer.batch_decode(input_ids)[0]
    return decoded_text, input_ids


def run_text_generation_tp_single_token(
):
    print(f"Running text generation for {MODEL_NAME}")

    # Setup environment
    setup_xla_environment()
    mesh = create_device_mesh()

    # Load model for text generation (not just hidden states)
    print("Loading model for text generation...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
    model = model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token


    print("Applying tensor parallel sharding to CausalLM model...")
    apply_tensor_parallel_sharding(model, mesh)

    print(f"\nInput prompt: {PROMPT}")
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt", padding=True, truncation=True)

    # Move inputs to XLA device and mark sharding
    input_ids = input_ids.to(torch_xla.device())
    
    xs.mark_sharding(input_ids, mesh, (None, None))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
    
    # Move back to CPU for decoding
    torch_xla.sync(True, True) # Wait until the output is ready
    generated_logits = outputs.logits.to("cpu")
    last_hidden_state = outputs.hidden_states[-1].to("cpu")

    next_token_logits = generated_logits[:, -1, :]  # Get logits for the last token
    next_token_id = next_token_logits.argmax(dim=-1)  # Get the most probable next token ID
    print("Raw next token ID: ", next_token_id)
    print("Decoded Next token ID: ", tokenizer.decode(next_token_id[0]))
    return last_hidden_state, next_token_id



def run_text_generation_tp_multi_token_with_cache(
    num_tokens: int = 64
):
    setup_xla_environment()
    mesh = create_device_mesh()
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Applying tensor parallel sharding to CausalLM model...")
    apply_tensor_parallel_sharding(model, mesh)

    input_ids = tokenizer.encode(PROMPT, return_tensors="pt", padding=True, truncation=True)
    past_key_values = None

    for step in range(num_tokens):
        print(f"Step {step + 1}/{num_tokens}")
        
        # For first step, use full input_ids; for subsequent steps, use only the last token
        current_input = input_ids if step == 0 else input_ids[:, -1:]

        # Generate next token with cache
        next_token_id, past_key_values = generate_single_token_with_cache(
            model, current_input, past_key_values, is_multi_device=True, mesh=mesh
        )
        
        if next_token_id == tokenizer.eos_token_id:
            break
            
        # Append next token to input_ids
        input_ids = torch.cat((input_ids, next_token_id), dim=1)

    decoded_text = tokenizer.batch_decode(input_ids)[0]
    return decoded_text, input_ids


def main():
    """Main function demonstrating tensor parallelism setup."""
    print("Torch-XLA Tensor Parallelism for Llama Models")
    print("=" * 50)

    try:
        # Demonstrate text generation with configurable max tokens
        print("\n🔤 TEXT GENERATION DEMO")
        print("-" * 30)
        
        # run_text_generation_cpu_multi_token()
        # text, output_ids = run_text_generation_tp_multi_token()
        text, output_ids = run_text_generation_tp_multi_token_with_cache(10)
        print(f"Generated text: {text}")
        print(f"Output IDs: {output_ids}")

        # def compute_pcc(x: torch.Tensor, y: torch.Tensor) -> float:
        #     """Compute Pearson Correlation Coefficient."""
        #     assert x.shape == y.shape, "Input tensors must have the same shape"
        #     x_flat, y_flat = x.flatten(), y.flatten()
        #     vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
        #     denom = vx.norm() * vy.norm()
        #     if denom == 0:
        #         return float("nan")
        #     return float((vx @ vy) / denom)
        
        # pcc = compute_pcc(cpu_hidden_state, tp_hidden_state)
        # print(f"Pearson Correlation Coefficient: {pcc:.6f}")
        # print("CPU Next Token ID: ", cpu_token_id)
        # print("TP Next Token ID: ", tp_token_id)

    except Exception as e:
        print(f"Error during execution: {e}")
        print("This might be due to missing dependencies or hardware requirements.")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
