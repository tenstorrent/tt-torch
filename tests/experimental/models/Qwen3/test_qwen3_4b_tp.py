# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/Qwen/Qwen2.5-1.5B
import torch
import pytest

# Load model directly
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth
import tt_mlir
from third_party.tt_forge_models.qwen_3.embedding.pytorch import (
    ModelLoader,
    ModelVariant,
)
import torch_xla.core.xla_model as xm

from tt_torch.tools.utils import (
    calculate_pcc,
)
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import torch_xla
from tests.utils import create_device_mesh


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs(dtype_override=torch.bfloat16)


@pytest.mark.usefixtures("use_xla_spmd_environment")
def test_qwen3_4b_tp(record_property):
    cc = CompilerConfig()

    cc.enable_consteval = True
    cc.consteval_parameters = True

    variant = ModelVariant.QWEN_3_EMBEDDING_4B
    loader = ModelLoader(variant=variant)
    model_info = loader.get_model_info(variant=variant)

    model = loader.load_model(dtype_override=torch.bfloat16)
    # model.layers = model.layers[:1]
    batch_size = 1
    max_length = 128
    input_ids = torch.randint(
        0, model.config.vocab_size, (batch_size, max_length), dtype=torch.int32
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.int32)
    cpu_output = model(
        input_ids=input_ids, attention_mask=attention_mask
    ).last_hidden_state

    model = model.to(torch_xla.device())
    mesh = create_device_mesh((1, xr.global_runtime_device_count()), ("batch", "model"))

    for layer in model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, ("batch", "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))

    input_ids = input_ids.to(torch_xla.device())
    attention_mask = attention_mask.to(torch_xla.device())
    output = model(input_ids=input_ids).last_hidden_state.to("cpu")
    pcc = calculate_pcc(cpu_output, output)
    print(f"PCC: {pcc}")
    # assert pcc > 0.99
