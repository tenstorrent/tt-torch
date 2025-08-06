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
from typing import Tuple
from transformers import LlamaModel, LlamaConfig


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

    Returns:
        Mesh object for SPMD operations
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


def apply_tensor_parallel_sharding(model: LlamaModel, mesh: Mesh) -> None:
    """
    Apply tensor parallel sharding to a Llama model.

    This function modifies the model in-place to add sharding annotations
    for tensor parallelism.
    """
    print("Applying tensor parallel sharding...")

    # Move model to XLA device first
    model = model.to(torch_xla.device())

    # Apply sharding to each transformer layer
    for i, layer in enumerate(model.layers):
        print(f"Sharding layer {i+1}/{len(model.layers)}")

        # ========================================
        # MLP (Feed-Forward) Layer Sharding
        # ========================================

        # Column parallel: Split output dimension across devices
        # up_proj: [hidden_size, intermediate_size] -> shard dim 0
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", "batch"))

        # gate_proj: [hidden_size, intermediate_size] -> shard dim 0
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", "batch"))

        # Row parallel: Split input dimension across devices
        # down_proj: [intermediate_size, hidden_size] -> shard dim 1
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, ("batch", "model"))

        # ========================================
        # Self-Attention Layer Sharding
        # ========================================

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

        # Note: LayerNorm parameters are typically replicated (small memory footprint)
        # You could also shard them if needed:
        # xs.mark_sharding(layer.input_layernorm.weight, mesh, (None,))
        # xs.mark_sharding(layer.post_attention_layernorm.weight, mesh, (None,))

    print("Tensor parallel sharding applied successfully!")


def prepare_inputs(mesh: Mesh, input_ids: torch.Tensor) -> torch.Tensor:
    print(
        f"Preparing inputs: batch_size={input_ids.shape[0]}, seq_length={input_ids.shape[1]}"
    )

    # Move to XLA device
    input_ids = input_ids.to(torch_xla.device())

    # Mark input sharding (typically replicated for inputs)
    xs.mark_sharding(input_ids, mesh, (None, None))

    return input_ids


def run_inference_comparison():
    """
    Run a complete example comparing single-device vs tensor-parallel inference.
    """
    model_name = "meta-llama/Meta-Llama-3.1-8B"
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
    print("Torch-XLA Tensor Parallelism for Llama Models")
    print("=" * 50)

    try:
        # Run the complete inference comparison
        pcc = run_inference_comparison()

        print("\n" + "=" * 50)
        print("Tensor parallelism demonstration completed successfully!")
        print(f"Final validation PCC: {pcc:.6f}")

    except Exception as e:
        print(f"Error during execution: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
