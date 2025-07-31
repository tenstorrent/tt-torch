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

PROMPT = "What is the name of the largest planet in our solar system?"

def setup_xla_environment():
    """Setup XLA environment for tensor parallelism."""
    print("Setting up XLA environment...")

    # Basic XLA configuration
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    os.environ["MESH_SHAPE"] = "1,8"

    # Initialize SPMD
    xr.use_spmd()
    torch_xla.sync(True, True)
    print("XLA environment configured.")


def create_device_mesh(
    num_devices: int = 8, mesh_shape: Tuple[int, int] = (1, 8)
) -> Mesh:
    """
    Create device mesh for tensor parallelism.

    Args:
        num_devices: Total number of devices
        mesh_shape: Shape of the device mesh (batch_dim, model_dim)

    Returns:
        Mesh object for SPMD operations
    """
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


def _apply_tensor_parallel_sharding_to_base_model(model: LlamaModel, mesh: Mesh) -> None:
    for i, layer in enumerate(model.layers):
        print(f"Sharding layer {i+1}/{len(model.layers)}")

        # Column parallel: Split output dimension across devices
        # up_proj: [hidden_size, intermediate_size] -> shard dim 0
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))

        # gate_proj: [hidden_size, intermediate_size] -> shard dim 0
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))

        # Row parallel: Split input dimension across devices
        # down_proj: [intermediate_size, hidden_size] -> shard dim 1
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

        # Column parallel: Split attention heads across devices
        # q_proj: [hidden_size, num_heads * head_dim] -> shard dim 0
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))

        # k_proj: [hidden_size, num_kv_heads * head_dim] -> shard dim 0
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))

        # v_proj: [hidden_size, num_kv_heads * head_dim] -> shard dim 0
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))

        # Row parallel: Collect results from all devices
        # o_proj: [num_heads * head_dim, hidden_size] -> shard dim 1
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

def apply_tensor_parallel_sharding(model: LlamaForCausalLM, mesh: Mesh) -> None:
    model = model.to(torch_xla.device())
    print("LlamaForCausalLM Model: ", model)
    # print("Applying tensor parallel sharding to LlamaForCausalLM...")
    # _apply_tensor_parallel_sharding(model, mesh)
    print("Applying tensor parallel sharding to LlamaModel...")
    _apply_tensor_parallel_sharding_to_base_model(model.model, mesh)

    # Also shard the language modeling head
    # print("LM Head weight shape: ", model.lm_head.weight.shape)
    xs.mark_sharding(model.lm_head.weight, mesh, ("model", None))

    print("Tensor parallel sharding applied successfully!")


def run_text_generation_on_cpu(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B"):
    llama = LlamaForCausalLM.from_pretrained(model_name)
    llama = llama.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(PROMPT, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    print("HET DEBUG - inputs:", inputs)

    with torch.no_grad():
        outputs = llama(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    cpu_logits = outputs.logits
    next_token_logits = cpu_logits[:, -1, :]  # Get logits for the last token
    next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
    print("Raw next token ID: ", next_token_id)
    print("Decoded Next token ID: ", tokenizer.decode(next_token_id[0]))

    return outputs.hidden_states[-1], next_token_id

def run_text_generation(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
):
    print(f"Running text generation for {model_name}")

    # Setup environment
    setup_xla_environment()
    mesh = create_device_mesh()

    # Load model for text generation (not just hidden states)
    print("Loading model for text generation...")
    model = LlamaForCausalLM.from_pretrained(model_name)
    model = model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    print("Applying tensor parallel sharding to CausalLM model...")
    apply_tensor_parallel_sharding(model, mesh)

    print(f"\nInput prompt: {PROMPT}")
    
    # Tokenize the prompt
    inputs = tokenizer(PROMPT, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Move inputs to XLA device and mark sharding
    input_ids = input_ids.to(torch_xla.device())
    attention_mask = attention_mask.to(torch_xla.device())
    
    xs.mark_sharding(input_ids, mesh, (None, None))
    xs.mark_sharding(attention_mask, mesh, (None, None))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    
    # Move back to CPU for decoding
    torch_xla.sync(True, True)
    generated_logits = outputs.logits.to("cpu")
    last_hidden_state = outputs.hidden_states[-1].to("cpu")

    next_token_logits = generated_logits[:, -1, :]  # Get logits for the last token
    next_token_id = next_token_logits.argmax(dim=-1)  # Get the most probable next token ID
    print("Raw next token ID: ", next_token_id)
    print("Decoded Next token ID: ", tokenizer.decode(next_token_id[0]))
    return last_hidden_state, next_token_id

def main():
    """Main function demonstrating tensor parallelism setup."""
    print("Torch-XLA Tensor Parallelism for Llama Models")
    print("=" * 50)

    try:
        # Demonstrate text generation with configurable max tokens
        print("\n🔤 TEXT GENERATION DEMO")
        print("-" * 30)
        
        cpu_hidden_state, cpu_token_id = run_text_generation_on_cpu()
        tp_hidden_state, tp_token_id = run_text_generation()
        # run_text_generation()

        def compute_pcc(x: torch.Tensor, y: torch.Tensor) -> float:
            """Compute Pearson Correlation Coefficient."""
            assert x.shape == y.shape, "Input tensors must have the same shape"
            x_flat, y_flat = x.flatten(), y.flatten()
            vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
            denom = vx.norm() * vy.norm()
            if denom == 0:
                return float("nan")
            return float((vx @ vy) / denom)
        
        pcc = compute_pcc(cpu_hidden_state, tp_hidden_state)
        print(f"Pearson Correlation Coefficient: {pcc:.6f}")
        print("CPU Next Token ID: ", cpu_token_id)
        print("TP Next Token ID: ", tp_token_id)

    except Exception as e:
        print(f"Error during execution: {e}")
        print("This might be due to missing dependencies or hardware requirements.")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
