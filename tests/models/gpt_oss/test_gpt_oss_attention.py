# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_torch

import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from transformers import AutoConfig

# Import the specific attention components from the model file
# Assuming the model file is available as a module or you can import from it
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssRotaryEmbedding,
    GptOssConfig,
)


class GPT_OSS_Attention:
    def __init__(self):
        self.model_name = "openai/gpt-oss-20b"
        self.layer_idx = 0  # Test first attention layer

    def _create_config(self):
        """Create a minimal config for attention testing"""
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

        # Override config for testing
        config.hidden_size = 2048  # Smaller for testing
        config.num_attention_heads = 32
        config.num_key_value_heads = 8  # GQA
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.attention_dropout = 0.0
        config.attention_bias = False
        config.sliding_window = None  # Full attention for simplicity
        config.max_position_embeddings = 2048
        config.rms_norm_eps = 1e-6
        config.layer_types = ["full_attention"]  # Single layer type for testing
        config._attn_implementation = "eager"  # Set explicit attention implementation

        return config

    def _create_attention_layer(self, config):
        """Create and initialize attention layer"""
        attention = GptOssAttention(config, layer_idx=self.layer_idx)
        rotary_emb = GptOssRotaryEmbedding(config)

        # Initialize weights and convert to bfloat16
        attention.eval()
        attention = attention.to(torch.bfloat16)
        rotary_emb = rotary_emb.to(torch.bfloat16)

        return attention, rotary_emb

    def _create_synthetic_inputs(self, config, batch_size=2, seq_len=128):
        """Create synthetic inputs for attention testing"""
        hidden_size = config.hidden_size

        # Use bfloat16 consistently
        dtype = torch.bfloat16

        # Create random hidden states
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

        # Create position IDs (keep as long for indexing)
        position_ids = (
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        )

        # Create causal attention mask - the model expects additive mask format
        # Create lower triangular mask (1s for allowed positions, 0s for masked)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=dtype))

        # Convert to additive format (0 for allowed, large negative for masked)
        attention_mask = torch.where(
            causal_mask.bool(),
            torch.tensor(0.0, dtype=dtype),
            torch.tensor(torch.finfo(dtype).min, dtype=dtype),
        )

        # Expand for batch and heads: [batch_size, num_heads, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = attention_mask.expand(
            batch_size, config.num_attention_heads, -1, -1
        )

        return hidden_states, position_ids, attention_mask


def run_attention_test():
    """
    Run a single forward pass through the attention layer using tensor parallelism.
    """
    print("Setting up GPT-OSS Attention Layer Test...")

    gpt_oss_attention = GPT_OSS_Attention()
    config = gpt_oss_attention._create_config()

    print(
        f"Config: hidden_size={config.hidden_size}, "
        f"num_heads={config.num_attention_heads}, "
        f"num_kv_heads={config.num_key_value_heads}"
    )

    print("Creating attention layer...")
    attention_layer, rotary_emb = gpt_oss_attention._create_attention_layer(config)

    print("Creating synthetic inputs...")
    batch_size, seq_len = 2, 128
    (
        hidden_states,
        position_ids,
        attention_mask,
    ) = gpt_oss_attention._create_synthetic_inputs(config, batch_size, seq_len)

    print(
        f"Input shapes: hidden_states={hidden_states.shape} ({hidden_states.dtype}), "
        f"position_ids={position_ids.shape} ({position_ids.dtype}), "
        f"attention_mask={attention_mask.shape} ({attention_mask.dtype})"
    )

    print("Moving inputs to XLA device...")
    device = torch_xla.device()
    hidden_states = hidden_states.to(device)
    position_ids = position_ids.to(device)
    attention_mask = attention_mask.to(device)

    print("Moving attention layer to XLA device...")
    attention_layer = attention_layer.to(device)
    rotary_emb = rotary_emb.to(device)

    print(
        f"After moving to XLA - hidden_states dtype: {hidden_states.dtype}, "
        f"attention_mask dtype: {attention_mask.dtype}"
    )

    print("Computing position embeddings...")
    with torch.no_grad():
        # Ensure position_ids are on the same device
        position_embeddings = rotary_emb(hidden_states, position_ids)
        cos, sin = position_embeddings

        # Ensure position embeddings are in bfloat16
        cos = cos.to(torch.bfloat16)
        sin = sin.to(torch.bfloat16)
        position_embeddings = (cos, sin)

        print(
            f"Position embeddings: cos={cos.shape} ({cos.dtype}), sin={sin.shape} ({sin.dtype})"
        )

    print("Running Attention Layer Forward Pass...")
    with torch.no_grad():
        try:
            # Forward pass through attention
            attn_output, attn_weights = attention_layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=None,
                cache_position=None,
            )

            print("Attention forward pass completed!")
            print(f"Output shapes: attn_output={attn_output.shape}")
            if attn_weights is not None:
                print(f"Attention weights shape: {attn_weights.shape}")

            # Synchronize XLA before moving to CPU
            print("Synchronizing XLA device...")
            xm.mark_step()
            xm.wait_device_ops()

            # Move results back to CPU for inspection
            print("Moving results to CPU...")
            attn_output_cpu = attn_output.cpu()

            # Print some basic statistics
            print(f"Attention output stats:")
            print(f"  Mean: {attn_output_cpu.mean():.6f}")
            print(f"  Std: {attn_output_cpu.std():.6f}")
            print(f"  Min: {attn_output_cpu.min():.6f}")
            print(f"  Max: {attn_output_cpu.max():.6f}")

        except Exception as e:
            print(f"Error during attention forward pass: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback

            traceback.print_exc()
            raise

    print("Attention layer test completed successfully!")


def main():
    print("GPT-OSS Attention Layer Test with Torch-XLA SPMD")
    print("=" * 60)

    try:
        run_attention_test()
        return 0
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
