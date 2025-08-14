# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_torch

import os
import torch
import pytest
from transformers import AutoModelForCausalLM, Qwen3Config
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np

from tt_torch.tools.utils import (
    calculate_pcc,
)

def setup_xla_environment():
    num_devices = xr.global_runtime_device_count()
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    os.environ["MESH_SHAPE"] = f"1,{num_devices}"
    
    xr.use_spmd()

def create_device_mesh() -> Mesh:
    num_devices = xr.global_runtime_device_count()
    print(f"Number of devices: {num_devices}") 
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices)) 
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Mesh shape: {mesh_shape}, Device IDs: {device_ids}")
    return mesh
    
def apply_tensor_parallel_sharding_causal(model: AutoModelForCausalLM, mesh: Mesh) -> None:
    model = model.to(torch_xla.device())
    for layer in model.model.layers:
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", "batch"))  
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", "batch")) 
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, ("batch", "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", "batch"))  
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))


@pytest.mark.parametrize(
    "data_type", [torch.bfloat16], ids=["bf16"]
)
@pytest.mark.parametrize("sequence_length", [32, 64, 128], ids=["32", "64", "128"])
def test_qwen3_embedding_eager(data_type, sequence_length):
    torch.manual_seed(42)

    setup_xla_environment()
    mesh = create_device_mesh()

    model_name = "Qwen/Qwen3-Embedding-8B"
    config = Qwen3Config.from_pretrained(model_name)
    config.num_hidden_layers = 18
    
    model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=data_type
    ).eval()

    batch_size = 1
    inputs = torch.randint(
        0, config.vocab_size, (batch_size, sequence_length), dtype=torch.int32
    )
    
    # Get CPU baseline
    outputs = model(inputs)
    cpu_outputs = outputs.logits

    # Apply tensor parallel sharding
    apply_tensor_parallel_sharding_causal(model, mesh)

    # Move inputs to XLA device and mark sharding
    inputs = inputs.to(torch_xla.device())
    xs.mark_sharding(inputs, mesh, (None, None)) # Replicate inputs to all devices

    outputs = model(inputs)
    torch_xla.sync(True, True)
    tt_outputs = outputs.logits.to("cpu")

    # Calculate PCC
    pcc = calculate_pcc(tt_outputs, cpu_outputs)
    print(f"PCC: {pcc}")
    
    assert pcc >= 0.90
