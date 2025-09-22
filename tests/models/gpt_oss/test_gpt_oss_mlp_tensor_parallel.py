# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_torch

import sys
import torch
import torch_xla
import os
import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
from transformers import AutoConfig
from torch import nn

# Import the specific MLP + RMSNorm
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssMLP,
    GptOssConfig,
    GptOssRMSNorm,
)

def calculate_pcc(tensor1, tensor2):
    """Compute PCC using torch.corrcoef"""
    # Flatten tensors to 1D
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    
    # Stack tensors and compute correlation matrix
    stacked = torch.stack([flat1, flat2])
    corr_matrix = torch.corrcoef(stacked)
    
    # PCC is the off-diagonal element
    pcc = corr_matrix[0, 1]
    return pcc.item()



def setup_xla_environment():
    """Setup XLA environment for tensor parallelism."""
    print("Setting up XLA environment...")

    # Converts the StableHLO emitted by torch-xla to the Shardy dialect.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    # Initialize SPMD
    xr.use_spmd()
    print("XLA environment configured.")


def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


class GPT_OSS_MLP:
    def __init__(self):
        self.model_name = "openai/gpt-oss-20b"
        self.layer_idx = 0

    def _create_config(self):
        """Create a minimal config for MLP testing"""
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

        # Override config for testing
        # config.hidden_size = 2048
        # config.intermediate_size = 8192
        # config.num_local_experts = 4
        # config.num_experts_per_tok = 2
        # config.rms_norm_eps = 1e-6
        # config.layer_types = ["full_attention"]

        return config

    def _create_mlp_layer(self, config):
        """Create and initialize MLP layer"""
        mlp = GptOssMLP(config).eval()
        mlp.router.weight = nn.Parameter(torch.randn(config.num_local_experts, config.hidden_size))
        mlp.router.bias = nn.Parameter(torch.randn(config.num_local_experts))

        mlp.experts.gate_up_proj = nn.Parameter(torch.randn(config.num_local_experts, config.hidden_size, 2 * config.intermediate_size))
        mlp.experts.gate_up_proj_bias = nn.Parameter(torch.randn(config.num_local_experts, 2 * config.intermediate_size))
        mlp.experts.down_proj = nn.Parameter(torch.randn(config.num_local_experts, config.intermediate_size, config.hidden_size))
        mlp.experts.down_proj_bias = nn.Parameter(torch.randn(config.num_local_experts, config.hidden_size))

        mlp = mlp.to(torch.bfloat16)

        return mlp

    def _create_normed_inputs(self, config, batch_size=1, seq_len=128):
        """Create synthetic hidden states and RMSNorm them"""
        dtype = torch.bfloat16
        hidden_states = torch.randn(
            batch_size, seq_len, config.hidden_size, dtype=dtype
        )

        # Apply RMSNorm just like in decoder layer
        rmsnorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hidden_states = rmsnorm(hidden_states.to(torch.float32)).to(dtype)

        return hidden_states

    def _setup_tensor_parallel(self, mlp_layer, mesh):
        """Setup tensor parallel sharding for MLP layer"""

        mlp_layer = mlp_layer.to(torch_xla.device())

        # Router weights - replicated (needed for routing decisions)
        xs.mark_sharding(mlp_layer.router.weight, mesh, (None, None))
        xs.mark_sharding(mlp_layer.router.bias, mesh, (None,))

        # Expert weights - optimal tensor parallelism configuration
        # Keep gate_up_proj replicated to avoid XLA sharding conflicts
        xs.mark_sharding(mlp_layer.experts.gate_up_proj, mesh, ("model", None, None))
        xs.mark_sharding(mlp_layer.experts.gate_up_proj_bias, mesh, ("model", None))

        # Shard down_proj for tensor parallelism (most beneficial for performance)
        # down_proj: (num_experts, intermediate_size, hidden_size)
        xs.mark_sharding(mlp_layer.experts.down_proj, mesh, ("model", None, None))
        xs.mark_sharding(mlp_layer.experts.down_proj_bias, mesh, ("model", None))

        return mlp_layer



def run_on_cpu():
    torch.manual_seed(42)
    gpt_oss_mlp = GPT_OSS_MLP()
    config = gpt_oss_mlp._create_config()


    mlp_layer = gpt_oss_mlp._create_mlp_layer(config)
    batch_size, seq_len = 1, 128
    hidden_states = gpt_oss_mlp._create_normed_inputs(config, batch_size, seq_len)
    print(
        f"Input shapes (after RMSNorm): hidden_states={hidden_states.shape} ({hidden_states.dtype})"
    )
    print("Running MLP layer forward pass on CPU...")
    cpu_mlp_output, cpu_router_scores = mlp_layer(hidden_states)
    print(f"CPU output: {cpu_mlp_output.shape} ({cpu_mlp_output.dtype})")
    print(f"CPU router scores: {cpu_router_scores.shape} ({cpu_router_scores.dtype})")
    print(f"CPU output full: {cpu_mlp_output}")
    print(f"CPU router scores full: {cpu_router_scores}")

    print(f"CPU MLP output stats:")
    print(f"  Mean: {cpu_mlp_output.mean()}")
    print(f"  Std: {cpu_mlp_output.std()}")
    print(f"  Min: {cpu_mlp_output.min()}")
    print(f"  Max: {cpu_mlp_output.max()}")
    print(f"CPU router scores stats:")
    print(f"  Mean: {cpu_router_scores.mean()}")
    print(f"  Std: {cpu_router_scores.std()}")
    print(f"  Min: {cpu_router_scores.min()}")
    print(f"  Max: {cpu_router_scores.max()}")


def compare_against_cpu():
    torch.manual_seed(42)
    setup_xla_environment()
    mesh = create_device_mesh()
    gpt_oss_mlp = GPT_OSS_MLP()
    config = gpt_oss_mlp._create_config()


    mlp_layer = gpt_oss_mlp._create_mlp_layer(config)
    batch_size, seq_len = 1, 128
    hidden_states = gpt_oss_mlp._create_normed_inputs(config, batch_size, seq_len)
    print(
        f"Input shapes (after RMSNorm): hidden_states={hidden_states.shape} ({hidden_states.dtype})"
    )
    print("Running MLP layer forward pass on CPU...")
    cpu_mlp_output, cpu_router_scores = mlp_layer(hidden_states)
    
    print("Running MLP layer forward pass on device...")
    mlp_layer = gpt_oss_mlp._setup_tensor_parallel(mlp_layer, mesh)
    hidden_states = hidden_states.to(torch_xla.device())
    hidden_states = xs.mark_sharding(hidden_states, mesh, (None, None, None))
    
    dev_mlp_output, dev_router_scores = mlp_layer(hidden_states)
    torch_xla.sync()
    dev_mlp_output = dev_mlp_output.to("cpu")
    dev_router_scores = dev_router_scores.to("cpu")
    print("Comparing results...")
    print(f"CPU output: {cpu_mlp_output.shape} ({cpu_mlp_output.dtype})")
    print(f"Device output: {dev_mlp_output.shape} ({dev_mlp_output.dtype})")
    print(f"CPU router scores: {cpu_router_scores.shape} ({cpu_router_scores.dtype})")
    print(f"Device router scores: {dev_router_scores.shape} ({dev_router_scores.dtype})")

    output_pcc = calculate_pcc(dev_mlp_output, cpu_mlp_output)
    router_scores_pcc = calculate_pcc(dev_router_scores, cpu_router_scores)
    print(f"CPU output full: {cpu_mlp_output}")
    print(f"Device output full: {dev_mlp_output}")
    print(f"CPU router scores full: {cpu_router_scores}")
    print(f"Device router scores full: {dev_router_scores}")
    print(f"Output PCC: {output_pcc}")
    print(f"Router scores PCC: {router_scores_pcc}")

    print()
    print()

    print(f"CPU MLP output stats:")
    print(f"  Mean: {cpu_mlp_output.mean()}")
    print(f"  Std: {cpu_mlp_output.std()}")
    print(f"  Min: {cpu_mlp_output.min()}")
    print(f"  Max: {cpu_mlp_output.max()}")
    print(f"CPU router scores stats:")
    print(f"  Mean: {cpu_router_scores.mean()}")
    print(f"  Std: {cpu_router_scores.std()}")
    print(f"  Min: {cpu_router_scores.min()}")
    print(f"  Max: {cpu_router_scores.max()}")

    print()
    print()

    print(f"Device MLP output stats:")
    print(f"  Mean: {dev_mlp_output.mean()}")
    print(f"  Std: {dev_mlp_output.std()}")
    print(f"  Min: {dev_mlp_output.min()}")
    print(f"  Max: {dev_mlp_output.max()}")
    print(f"Device router scores stats:")
    print(f"  Mean: {dev_router_scores.mean()}")
    print(f"  Std: {dev_router_scores.std()}")
    print(f"  Min: {dev_router_scores.min()}")
    print(f"  Max: {dev_router_scores.max()}")



def run_mlp_test():
    """
    Run a single forward pass through the MLP layer using tensor parallelism.
    """
    print("Setting up GPT-OSS Multichip MLP Layer Test...")

    # Setup environment and mesh
    setup_xla_environment()
    mesh = create_device_mesh()

    gpt_oss_mlp = GPT_OSS_MLP()
    config = gpt_oss_mlp._create_config()

    print(
        f"Config: hidden_size={config.hidden_size}, "
        f"intermediate_size={config.intermediate_size}, "
        f"num_experts={config.num_local_experts}, "
        f"top_k={config.num_experts_per_tok}"
    )

    print("Creating MLP layer...")
    mlp_layer = gpt_oss_mlp._create_mlp_layer(config)

    print("Setting up tensor parallelism...")
    mlp_layer = gpt_oss_mlp._setup_tensor_parallel(mlp_layer, mesh)

    print("Creating normalized synthetic inputs...")
    batch_size, seq_len = 1, 128
    hidden_states = gpt_oss_mlp._create_normed_inputs(config, batch_size, seq_len)

    print(
        f"Input shapes (after RMSNorm): hidden_states={hidden_states.shape} ({hidden_states.dtype})"
    )

    print("Moving inputs to XLA device...")
    device = torch_xla.device()

    # Mark input sharding - inputs are replicated across all devices
    hidden_states = hidden_states.to(device)
    hidden_states = xs.mark_sharding(hidden_states, mesh, (None, None, None))

    print(f"After moving to XLA - hidden_states dtype: {hidden_states.dtype}")

    print("Running MLP Layer Forward Pass with Tensor Parallelism...")
    with torch.no_grad():
        try:
            mlp_output, router_scores = mlp_layer(hidden_states)

            print("MLP forward pass completed!")
            print(
                f"Output shapes: mlp_output={mlp_output.shape}, router_scores={router_scores.shape}"
            )

            print("Synchronizing XLA device...")
            torch_xla.sync()

            print("Moving results to CPU...")
            mlp_output = mlp_output.to("cpu")
            router_scores = router_scores.to("cpu")

            # Stats
            print(f"MLP output stats:")
            print(f"  Mean: {mlp_output.mean()}")
            print(f"  Std: {mlp_output.std()}")
            print(f"  Min: {mlp_output.min()}")
            print(f"  Max: {mlp_output.max()}")

            print(f"Router scores stats:")
            print(f"  Mean: {router_scores.mean()}")
            print(f"  Std: {router_scores.std()}")
            print(f"  Min: {router_scores.min()}")
            print(f"  Max: {router_scores.max()}")

        except Exception as e:
            print(f"Error during MLP forward pass: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback

            traceback.print_exc()
            raise

    print("MLP layer tensor parallel test completed successfully!")


def main():
    print("GPT-OSS MLP Layer Test with Torch-XLA SPMD Tensor Parallelism")
    print("=" * 60)

    try:
        run_mlp_test()
        # compare_against_cpu()
        # run_on_cpu()
        return 0
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
