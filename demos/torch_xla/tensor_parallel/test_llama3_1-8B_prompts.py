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


def _apply_tensor_parallel_sharding(model: LlamaModel, mesh: Mesh) -> None:
    # model = model.to(torch_xla.device())
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
    _apply_tensor_parallel_sharding(model.model, mesh)

    # Also shard the language modeling head
    print("LM Head weight shape: ", model.lm_head.weight.shape)
    xs.mark_sharding(model.lm_head.weight, mesh, ("model", None))

    print("Tensor parallel sharding applied successfully!")


def prepare_inputs(mesh: Mesh, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Prepare input tensors with appropriate sharding.

    Args:
        config: Model configuration
        mesh: Device mesh
        batch_size: Batch size
        seq_length: Sequence length

    Returns:
        Sharded input tensor
    """
    print(
        f"Preparing inputs: batch_size={input_ids.shape[0]}, seq_length={input_ids.shape[1]}"
    )

    # Move to XLA device
    input_ids = input_ids.to(torch_xla.device())

    # Mark input sharding (typically replicated for inputs)
    xs.mark_sharding(input_ids, mesh, (None, None))

    return input_ids


def run_text_generation_on_cpu(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B"):
    llama = LlamaForCausalLM.from_pretrained(model_name)
    # llama = llama.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(PROMPT, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    print("HET DEBUG - inputs:", inputs)

    with torch.no_grad():
        outputs = llama(input_ids=input_ids, attention_mask=attention_mask)
    cpu_logits = outputs.logits
    cpu_predictions = cpu_logits.argmax(dim=-1)
    print("CPU Predictions: ", [tokenizer.decode(token_id) for token_id in cpu_predictions])

def run_text_generation(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
):
    """
    Run text generation with the Llama model.

    Args:
        model_name: HuggingFace model name to load
    """
    print(f"Running text generation for {model_name}")

    # Setup environment
    setup_xla_environment()
    mesh = create_device_mesh()

    # Load model for text generation (not just hidden states)
    print("Loading model for text generation...")
    # config = LlamaConfig.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name
    )
    model = model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # ========================================
    # Apply Tensor Parallelism to CausalLM Model
    # ========================================
    print("Applying tensor parallel sharding to CausalLM model...")
    
    # Apply sharding to the base model (model.model is the LlamaModel inside LlamaForCausalLM)
    apply_tensor_parallel_sharding(model, mesh)

    # ========================================
    # Prepare Input
    # ========================================
    print(f"\nInput prompt: {PROMPT}")
    
    # Tokenize the prompt
    inputs = tokenizer(PROMPT, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    print(f"Input IDs: {input_ids}")
    attention_mask = inputs["attention_mask"]
    print(f"Attention Mask: {attention_mask}")
    
    print(f"Input token length: {input_ids.shape[1]}")
    
    # Move to XLA device and mark sharding
    input_ids = input_ids.to(torch_xla.device())
    attention_mask = attention_mask.to(torch_xla.device())
    
    xs.mark_sharding(input_ids, mesh, (None, None))
    xs.mark_sharding(attention_mask, mesh, (None, None))

    # ========================================
    # Generate Text
    # ========================================
    
    # with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=GenerationConfig(
            num_beams=1,
            do_sample=False,
            max_new_tokens=64,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id,
            temperature=0.0,
            top_p=1.0
        )
    )
        # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # generated_ids = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     temperature=temperature,
        #     do_sample=do_sample,
        #     pad_token_id=tokenizer.eos_token_id,
        #     eos_token_id=tokenizer.eos_token_id,
        # )
    
    # Move back to CPU for decoding
    print("Outputs: ", outputs)
    generated_ids = outputs.to("cpu")
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # generated_logits = outputs.logits.to("cpu")
    # next_token_ids = generated_logits[0].argmax(dim=-1)
    print("Generated text: ", generated_texts[0])
    
    # # Decode the generated text
    # generated_text = tokenizer.decode(next_token_ids[0], skip_special_tokens=True)

    # # Extract only the newly generated part
    # original_text = tokenizer.decode(input_ids[0].cpu(), skip_special_tokens=True)
    # new_text = generated_text[len(original_text):].strip()
    
    # print(f"\n{'='*50}")
    # print("GENERATED RESPONSE:")
    # print(f"{'='*50}")
    # print(new_text)
    # print(f"{'='*50}")
    # print(f"Total tokens generated: {generated_ids.shape[1] - input_ids.shape[1]}")
    
    # return generated_text, new_text


def run_inference_comparison(model_name: str = "meta-llama/Meta-Llama-3.1-8B"):
    """
    Run a complete example comparing single-device vs tensor-parallel inference.

    Args:
        model_name: HuggingFace model name to load
    """
    print(f"Running inference comparison for {model_name}")

    # Setup environment
    setup_xla_environment()
    mesh = create_device_mesh()

    # Load model and configuration
    print("Loading model...")
    config = LlamaConfig.from_pretrained(model_name)
    model = LlamaModel(config)

    # ========================================
    # Single Device Reference Run
    # ========================================
    print("\n=== Single Device Reference ===")

    # Prepare inputs for CPU/single device
    batch_size, seq_length = 1, 512
    input_ids_cpu = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    # Run on CPU for reference
    with torch.no_grad():
        model_cpu = model.cpu()
        outputs_cpu = model_cpu(input_ids=input_ids_cpu)
        reference_output = outputs_cpu.last_hidden_state

    print(f"CPU output shape: {reference_output.shape}")

    # ========================================
    # Tensor Parallel Run
    # ========================================
    print("\n=== Tensor Parallel Inference ===")

    # Apply tensor parallelism
    apply_tensor_parallel_sharding(model, mesh)

    # Prepare inputs for tensor parallel execution
    input_ids_tp = prepare_inputs(mesh, input_ids_cpu)

    # Run tensor parallel inference
    with torch.no_grad():
        outputs_tp = model(input_ids=input_ids_tp)
        tp_output = outputs_tp.last_hidden_state.cpu()

    print(f"Tensor parallel output shape: {tp_output.shape}")

    # ========================================
    # Validation
    # ========================================
    print("\n=== Validation ===")

    def compute_pcc(x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Pearson Correlation Coefficient."""
        assert x.shape == y.shape, "Input tensors must have the same shape"
        x_flat, y_flat = x.flatten(), y.flatten()
        vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
        denom = vx.norm() * vy.norm()
        if denom == 0:
            return float("nan")
        return float((vx @ vy) / denom)

    # Compare outputs
    pcc = compute_pcc(reference_output, tp_output)
    print(f"Pearson Correlation Coefficient: {pcc:.6f}")

    # Check if outputs are sufficiently similar
    if pcc > 0.90:
        print("✅ Tensor parallel implementation is correct!")
    else:
        print("❌ Tensor parallel outputs differ significantly from reference")
        print("This might indicate an implementation issue")

    return pcc


def main():
    """Main function demonstrating tensor parallelism setup."""
    print("Torch-XLA Tensor Parallelism for Llama Models")
    print("=" * 50)

    try:
        # Demonstrate text generation with configurable max tokens
        print("\n🔤 TEXT GENERATION DEMO")
        print("-" * 30)
        
        # You can adjust these parameters:
        # MAX_NEW_TOKENS = 100  # Change this to control response length
        # TEMPERATURE = 0.7     # Controls randomness (0.1 = more deterministic, 1.0 = more random)
        
        run_text_generation()
        # run_text_generation_on_cpu()
        
        # print(f"\n✅ Text generation completed!")
        # print(f"Generated {len(new_text.split())} words with max_new_tokens={MAX_NEW_TOKENS}")

        # Optional: Run the original inference comparison
        # print(f"\n\n🔍 INFERENCE COMPARISON DEMO")
        # print("-" * 30)
        # print("(Comparing single-device vs tensor-parallel hidden state outputs)")
        
        # pcc = run_inference_comparison()

        # print("\n" + "=" * 50)
        # print("All demonstrations completed successfully!")
        # print(f"Text generation max tokens: {MAX_NEW_TOKENS}")
        # print(f"Inference validation PCC: {pcc:.6f}")

    except Exception as e:
        print(f"Error during execution: {e}")
        print("This might be due to missing dependencies or hardware requirements.")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
