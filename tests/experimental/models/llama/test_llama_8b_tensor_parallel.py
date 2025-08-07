# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_torch

import os
import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaModel
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np

from tt_torch.tools.utils import (
    calculate_pcc,
)

def setup_xla_environment():
    # Basic XLA configuration
    num_devices = xr.global_runtime_device_count()
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    os.environ["MESH_SHAPE"] = f"1,{num_devices}"

    xr.use_spmd()

def create_device_mesh() -> Mesh:
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    return mesh


def apply_tensor_parallel_sharding_base(
    model: LlamaModel, mesh: Mesh, move_to_device: bool = True
) -> None:
    if move_to_device:
        model = model.to(torch_xla.device())

    # Apply sharding to each transformer layer
    for layer in model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, ("batch", "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))


def apply_tensor_parallel_sharding_causal(model: AutoModelForCausalLM, mesh: Mesh) -> None:
    model = model.to(torch_xla.device())
    apply_tensor_parallel_sharding_base(model.model, mesh, move_to_device=False)
    xs.mark_sharding(model.lm_head.weight, mesh, ("model", "batch"))

@pytest.mark.parametrize(
    "run_causal",
    [True, False],
    ids=["causal", "base"],
)
@pytest.mark.parametrize(
    "data_type", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"]
)
@pytest.mark.parametrize("sequence_length", [128, 256, 512], ids=["128", "256", "512"])
def test_llama_8b_eager(run_causal, data_type, sequence_length):
    torch.manual_seed(42)

    setup_xla_environment()
    mesh = create_device_mesh()

    model_name = "meta-llama/Llama-3.1-8B"
    config = LlamaConfig.from_pretrained(model_name)
    if run_causal:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=data_type
        ).eval()
    else:
        model = LlamaModel.from_pretrained(
            model_name, config=config, torch_dtype=data_type)

    batch_size = 1
    inputs = torch.randint(
        0, config.vocab_size, (batch_size, sequence_length), dtype=torch.int32
    )
    outputs = model(inputs)
    if run_causal:
        cpu_outputs = outputs.logits
    else:
        cpu_outputs = outputs.last_hidden_state

    if run_causal:
        apply_tensor_parallel_sharding_causal(model, mesh)
    else:
        apply_tensor_parallel_sharding_base(model, mesh)

    inputs = inputs.to(torch_xla.device())
    xs.mark_sharding(inputs, mesh, (None, None))

    outputs = model(inputs)
    # torch_xla.sync(True, True)
    if run_causal:
        tt_outputs = outputs.logits.to("cpu")
    else:
        tt_outputs = outputs.last_hidden_state.to("cpu")

    pcc = calculate_pcc(tt_outputs, cpu_outputs)
    print(f"PCC: {pcc}")
    assert pcc >= 0.96
