# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_torch

import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from transformers import AutoConfig

# Import the specific MLP + RMSNorm
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssMLP,
    GptOssConfig,
    GptOssRMSNorm,
)


class GPT_OSS_MLP:
    def __init__(self):
        self.model_name = "openai/gpt-oss-20b"
        self.layer_idx = 0

    def _create_config(self):
        """Create a minimal config for MLP testing"""
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

        # Override config for testing
        config.hidden_size = 2048
        config.intermediate_size = 8192
        config.num_local_experts = 4
        config.num_experts_per_tok = 2
        config.rms_norm_eps = 1e-6
        config.layer_types = ["full_attention"]

        return config

    def _create_mlp_layer(self, config):
        """Create and initialize MLP layer"""
        mlp = GptOssMLP(config)

        mlp.eval()
        mlp = mlp.to(torch.bfloat16)

        return mlp

    def _create_normed_inputs(self, config, batch_size=2, seq_len=128):
        """Create synthetic hidden states and RMSNorm them"""
        dtype = torch.bfloat16
        hidden_states = torch.randn(
            batch_size, seq_len, config.hidden_size, dtype=dtype
        )

        # Apply RMSNorm just like in decoder layer
        rmsnorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hidden_states = rmsnorm(hidden_states.to(torch.float32)).to(dtype)

        return hidden_states


def run_mlp_test():
    """
    Run a single forward pass through the MLP layer using tensor parallelism.
    """
    print("Setting up GPT-OSS MLP Layer Test...")

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

    print("Creating normalized synthetic inputs...")
    batch_size, seq_len = 2, 128
    hidden_states = gpt_oss_mlp._create_normed_inputs(config, batch_size, seq_len)

    print(
        f"Input shapes (after RMSNorm): hidden_states={hidden_states.shape} ({hidden_states.dtype})"
    )

    print("Moving inputs to XLA device...")
    device = torch_xla.device()
    hidden_states = hidden_states.to(device)

    print("Moving MLP layer to XLA device...")
    mlp_layer = mlp_layer.to(device)

    print(f"After moving to XLA - hidden_states dtype: {hidden_states.dtype}")

    print("Running MLP Layer Forward Pass...")
    with torch.no_grad():
        try:
            mlp_output, router_scores = mlp_layer(hidden_states)

            print("MLP forward pass completed!")
            print(
                f"Output shapes: mlp_output={mlp_output.shape}, router_scores={router_scores.shape}"
            )

            print("Synchronizing XLA device...")
            xm.mark_step()
            xm.wait_device_ops()

            print("Moving results to CPU...")
            mlp_output_cpu = mlp_output.cpu()
            router_scores_cpu = router_scores.cpu()

            # Stats
            print(f"MLP output stats:")
            print(f"  Mean: {mlp_output_cpu.mean():.6f}")
            print(f"  Std: {mlp_output_cpu.std():.6f}")
            print(f"  Min: {mlp_output_cpu.min():.6f}")
            print(f"  Max: {mlp_output_cpu.max():.6f}")

            print(f"Router scores stats:")
            print(f"  Mean: {router_scores_cpu.mean():.6f}")
            print(f"  Std: {router_scores_cpu.std():.6f}")
            print(f"  Min: {router_scores_cpu.min():.6f}")
            print(f"  Max: {router_scores_cpu.max():.6f}")

        except Exception as e:
            print(f"Error during MLP forward pass: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback

            traceback.print_exc()
            raise

    print("MLP layer test completed successfully!")


def main():
    print("GPT-OSS MLP Layer Test with Torch-XLA SPMD")
    print("=" * 60)

    try:
        run_mlp_test()
        return 0
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
