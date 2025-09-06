# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import tt_torch

import os, time
print(f'pid = {os.getpid()}')
time.sleep(3)  # Time to attach debugger if needed
import sys
import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
import torch_xla.core.xla_model as xm
from torch_xla.distributed.spmd import Mesh
import numpy as np
from transformers import AutoConfig

from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssRotaryEmbedding,
    GptOssConfig,
)

import copy
import math


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


def apply_attention_tensor_parallel_sharding(
    attention_layer: GptOssAttention, mesh: Mesh
) -> None:
    """
    Apply tensor parallel sharding to the GPT-OSS attention layer.
    """
    # Move attention layer to XLA device first
    attention_layer = attention_layer.to(torch_xla.device())

    print("Applying tensor parallel sharding to attention layer...")

    # ========================================
    # Self-Attention Layer Sharding - shard the heads across all devices
    # ========================================

    # q_proj: [num_heads * head_dim, hidden_size] -> colwise
    xs.mark_sharding(attention_layer.q_proj.weight, mesh, ("model", "batch"))
    print(f"Sharded q_proj: {attention_layer.q_proj.weight.shape}")

    # k_proj: [num_kv_heads * head_dim, hidden_size] -> colwise
    xs.mark_sharding(attention_layer.k_proj.weight, mesh, ("model", "batch"))
    print(f"Sharded k_proj: {attention_layer.k_proj.weight.shape}")

    # v_proj: [num_kv_heads * head_dim, hidden_size] -> colwise
    xs.mark_sharding(attention_layer.v_proj.weight, mesh, ("model", "batch"))
    print(f"Sharded v_proj: {attention_layer.v_proj.weight.shape}")

    # o_proj: [hidden_size, num_heads * head_dim] -> rowwise
    xs.mark_sharding(attention_layer.o_proj.weight, mesh, ("batch", "model"))
    print(f"Sharded o_proj: {attention_layer.o_proj.weight.shape}")

    # sinks -> local. rowwise
    if hasattr(attention_layer, "sinks") and attention_layer.sinks is not None:
        xs.mark_sharding(attention_layer.sinks, mesh, (None, ))
        print(f"Sharded sinks: {attention_layer.sinks.shape}")

    print("Tensor parallel sharding applied successfully to attention layer!")


def compare_tensors(a: torch.Tensor, b: torch.Tensor, name="attn_output",
                    atol=8e-3, rtol=1e-2, pcc_min: float | None = None) -> bool:
    """
    Prints max/mean abs diff, max relative diff, and PCC between a and b.
    Returns True if (max_abs <= atol or max_rel <= rtol) and, if pcc_min is set,
    PCC >= pcc_min (and not NaN). Otherwise returns False.
    """
    # bring to CPU FP32 for fair comparison
    a32 = a.detach().float().cpu()
    b32 = b.detach().float().cpu()

    diff = (a32 - b32).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = torch.maximum(a32.abs(), b32.abs())
    max_rel = (diff / torch.clamp(denom, min=1e-6)).max().item()

    # PCC (Pearson) on flattened vectors
    a1 = a32.reshape(-1)
    b1 = b32.reshape(-1)
    ac = a1 - a1.mean()
    bc = b1 - b1.mean()
    num = (ac * bc).sum()
    den = torch.sqrt((ac * ac).sum() * (bc * bc).sum())
    if den.item() == 0.0:
        pcc = 1.0 if torch.allclose(a1, b1, atol=atol, rtol=rtol) else float("nan")
    else:
        pcc = (num / den).item()

    print(f"[{name}] max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}, max_rel={max_rel:.6f}, PCC={pcc:.6f}")

    ok = (max_abs <= atol) or (max_rel <= rtol)
    if pcc_min is not None:
        ok = ok and (not math.isnan(pcc) and pcc >= pcc_min)
        print(f"[{name}] PCC check (>= {pcc_min}): {'OK' if (not math.isnan(pcc) and pcc >= pcc_min) else 'FAIL'}")

    print(f"[{name}] {'OK ✅' if ok else 'MISMATCH ❌'}  (atol={atol}, rtol={rtol}{', pcc_min='+str(pcc_min) if pcc_min is not None else ''})")
    return ok

def test_run_multichip_attention_test():
    print("Setting up GPT-OSS Multichip Attention Layer Test...")

    torch.manual_seed(0)
    np.random.seed(0)

    # Setup environment and mesh
    setup_xla_environment()
    mesh = create_device_mesh()

    gpt_oss_attention = GPT_OSS_Attention()
    config = gpt_oss_attention._create_config()
    print(f"Config: hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}, num_kv_heads={config.num_key_value_heads}")

    print("Creating attention layer...")
    attention_layer, rotary_emb = gpt_oss_attention._create_attention_layer(config)

    attention_cpu = copy.deepcopy(attention_layer).to("cpu").eval().to(torch.float32)
    rotary_cpu = copy.deepcopy(rotary_emb).to("cpu").eval().to(torch.float32)

    print("Creating synthetic inputs...")
    batch_size, seq_len = 1, 128
    hidden_states, position_ids, attention_mask = gpt_oss_attention._create_synthetic_inputs(config, batch_size, seq_len)

    with torch.no_grad():
        hs_for_rope = hidden_states.float()  # FP32 for CPU rotary
        cos_cpu, sin_cpu = rotary_cpu(hs_for_rope, position_ids)  
        pos_emb_cpu = (cos_cpu, sin_cpu)  
        pos_emb_xla = (cos_cpu.to(torch.bfloat16), sin_cpu.to(torch.bfloat16))

    print("Running CPU reference forward pass...")
    with torch.no_grad():
        attn_out_cpu, attn_w_cpu = attention_cpu(
            hidden_states=hidden_states.float(),          
            position_embeddings=pos_emb_cpu,              
            attention_mask=attention_mask.float(),        
            past_key_values=None,
            cache_position=None,
        )
    print(f"[CPU] attn_out={attn_out_cpu.shape}, dtype={attn_out_cpu.dtype}")

    print("Applying tensor parallel sharding...")
    apply_attention_tensor_parallel_sharding(attention_layer, mesh)

    print("Preparing inputs for tensor parallelism...")
    hidden_states_xla = hidden_states.to(torch.bfloat16).to(torch_xla.device())
    position_ids_xla = position_ids.to(torch_xla.device())
    attention_mask_xla = attention_mask.to(torch.bfloat16).to(torch_xla.device())
    xs.mark_sharding(attention_mask_xla, mesh, (None, "model", None, None))

    rotary_emb = rotary_emb.to(torch_xla.device())  

    with torch.no_grad():
        attn_out_xla, attn_w_xla = attention_layer(
            hidden_states=hidden_states_xla,
            position_embeddings=(pos_emb_xla[0].to(torch_xla.device()),
                                 pos_emb_xla[1].to(torch_xla.device())),
            attention_mask=attention_mask_xla,
            past_key_values=None,
            cache_position=None,
        )
    print(f"[XLA] attn_out={attn_out_xla.shape}, dtype={attn_out_xla.dtype}")

    attn_out_xla_cpu = attn_out_xla.detach().float().cpu()
    ok = compare_tensors(attn_out_xla_cpu, attn_out_cpu, name="attn_output",
                         atol=1e-2, rtol=1e-2, pcc_min=0.98)

    if not ok:
        diff = (attn_out_xla_cpu - attn_out_cpu).abs()
        print(f"percentiles | p50={diff.quantile(0.5).item():.6f} "
              f"p90={diff.quantile(0.9).item():.6f} p99={diff.quantile(0.99).item():.6f}")

    print("Done.")


def main():
    print("GPT-OSS Multichip Attention Layer Test with Torch-XLA SPMD")
    print("=" * 70)

    try:
        test_run_multichip_attention_test()
        return 0
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
