# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This import is unused but needed to register the Tenstorrent PJRT device with XLA
import tt_torch

import torch
import pytest
from transformers import GptOssForCausalLM, GptOssModel, GptOssConfig
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
from tests.utils import create_device_mesh

from tt_torch.tools.utils import (
    calculate_pcc,
)


def apply_tensor_parallel_sharding_base(
    model: GptOssModel, mesh: Mesh, move_to_device: bool = True
) -> None:
    if move_to_device:
        model = model.to(torch_xla.device())

    xs.mark_sharding(model.embed_tokens.weight, mesh, ("model", None))
    # Apply sharding to each transformer layer
    for layer in model.layers:
        # ========================================
        # MLP (Feed-Forward) Layer Sharding - shard the intermediate_size across devices
        # ========================================

        # EP try

        # Replicate all router matrices
        xs.mark_sharding(layer.mlp.router.weight, mesh, (None, None)) # [32, 2880]
        xs.mark_sharding(layer.mlp.router.bias, mesh, (None,)) # [32]

        # Shard all expert matrices on the experts dimension (dim 0)
        # [32, 2880, 5760]
        xs.mark_sharding(layer.mlp.experts.gate_up_proj, mesh, ("model", None, None))
        # [32, 5760]
        xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, ("model", None))
        # [32, 2880, 2880]
        xs.mark_sharding(layer.mlp.experts.down_proj, mesh, ("model", None, None))
        # [32, 2880]
        xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, ("model", None))

        # ========================================
        # Self-Attention Layer Sharding - shard the heads across all devices
        # ========================================

        # q_proj: [num_heads * head_dim, hidden_size] -> shard dim 0
        # [4096, 2880]
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.q_proj.bias, mesh, ("model",))

        # k_proj: [num_kv_heads * head_dim, hidden_size] -> shard dim 0
        # [512, 2880]
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.bias, mesh, ("model",))

        # v_proj: [num_kv_heads * head_dim, hidden_size] -> shard dim 0
        # [512, 2880]
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.bias, mesh, ("model",))

        # o_proj: [hidden_size, num_heads * head_dim] -> shard dim 1
        # [2880, 4096]
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))
        xs.mark_sharding(layer.self_attn.o_proj.bias, mesh, (None,))

        # sinks: [num_heads] -> shard dim 0
        # [64]
        if hasattr(layer.self_attn, "sinks") and layer.self_attn.sinks is not None:
            xs.mark_sharding(layer.self_attn.sinks, mesh, ("model", ))
            print(f"Sharded sinks: {layer.self_attn.sinks.shape}")


def apply_tensor_parallel_sharding_causal(
    model: GptOssForCausalLM, mesh: Mesh
) -> None:
    model = model.to(torch_xla.device())
    apply_tensor_parallel_sharding_base(model.model, mesh, move_to_device=False)
    xs.mark_sharding(model.lm_head.weight, mesh, ("model", None))


@pytest.mark.parametrize(
    "run_causal",
    [True],
    ids=["causal"],
)
@pytest.mark.parametrize("sequence_length", [128], ids=["128"])
@pytest.mark.usefixtures("use_xla_spmd_environment")
def test_gpt_oss_20b_eager(run_causal, sequence_length):
    torch.manual_seed(42)

    mesh = create_device_mesh((1, xr.global_runtime_device_count()), ("batch", "model"))

    model_name = "openai/gpt-oss-20b"
    config = GptOssConfig.from_pretrained(model_name)
    delattr(config, "quantization_config")
    config.use_cache = False
    if run_causal:
        model = GptOssForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=torch.bfloat16
        ).eval()
    else:
        model = GptOssModel.from_pretrained(
            model_name, config=config, torch_dtype=torch.bfloat16
        )

    batch_size = 1
    inputs = torch.randint(
        0, config.vocab_size, (batch_size, sequence_length), dtype=torch.int32
    )
    # Run model on CPU first
    outputs = model(inputs)
    if run_causal:
        cpu_outputs = outputs.logits
    else:
        cpu_outputs = outputs.last_hidden_state

    # Now run on devices
    if run_causal:
        apply_tensor_parallel_sharding_causal(model, mesh)
    else:
        apply_tensor_parallel_sharding_base(model, mesh)

    inputs = inputs.to(torch_xla.device())
    xs.mark_sharding(inputs, mesh, (None, None))  # Replicate inputs to all devices

    outputs = model(inputs)
    torch_xla.sync()  # Wait until all computations have finished
    if run_causal:
        tt_outputs = outputs.logits.to("cpu")
    else:
        tt_outputs = outputs.last_hidden_state.to("cpu")

    pcc = calculate_pcc(tt_outputs, cpu_outputs)
    print(f"PCC: {pcc}")
    # TODO: PCC is low due to experimental reduce_scatter - https://github.com/tenstorrent/tt-torch/issues/1209
    assert pcc >= 0.8 # PCC: -0.00028
