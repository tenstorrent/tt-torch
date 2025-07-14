# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tt_torch.dynamo.backend import BackendOptions


class ThisTester(ModelTester):
    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return model

    def _load_inputs(self):
        self.test_input = "This is a sample text from "
        inputs = self.tokenizer.encode_plus(
            self.test_input,
            return_tensors="pt",
            max_length=32,
            padding="max_length",
            truncation=True,
        )
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize("model_name", ["meta-llama/Llama-3.2-3B"])
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_llama_3b(record_property, model_name, mode, op_by_op):
    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    cc.enable_consteval = True

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        assert_atol=False,
        assert_pcc=True,
        record_property_handle=record_property,
    )
    results = tester.test_model()
    tester.finalize()


import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding, LlamaMLP, LlamaDecoderLayer, LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers import StaticCache
from tt_torch.tools.utils import calculate_pcc
import os
import torch_xla
import torch_xla.core.xla_model as xm

os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"
def setup_tt_environment():
    """Setup TensorTrent environment and plugin."""
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["MESH_SHAPE"] = "2,4"
    os.environ["LOGGER_LEVEL"] = "DEBUG"

    xr.use_spmd()
    torch_xla.sync(True, True)

def create_mesh():
    """Create device mesh for testing."""
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, 8)
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, ("batch", "model"))

def test_multichip():
    setup_tt_environment()
    mesh = create_mesh()
    B = 1
    S = 1024
    config = LlamaConfig.from_pretrained("meta-llama/Llama-3.2-3B")
    # config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-70B")
    config.num_hidden_layers = 1
    llama = LlamaModel(config).eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S))
    out_cpu = llama(input_ids=input_ids, attention_mask=None)

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = False
    cc.mesh = mesh

    options = BackendOptions()
    options.compiler_config = cc
    llama = torch.compile(llama, backend="tt", options=options)
    llama = llama.to(torch.bfloat16)
    # input_ids = input_ids.to("xla")
    # llama = llama.to("xla")

    static_cache = StaticCache(
        config=config,
        max_batch_size=B,
        max_cache_len=S,
        device="cpu",
        dtype=torch.bfloat16,
    )
    cache_position = torch.arange(0, S)
    # cache_position = cache_position.to("xla")

    for i, (key, value) in enumerate(zip(static_cache.key_cache, static_cache.value_cache)):
        static_cache.key_cache[i].shard_spec = (None, "model", None, None)
        static_cache.value_cache[i].shard_spec = (None, "model", None, None)

    for layer in llama.layers:
        layer.mlp.up_proj.weight.shard_spec = ("model", None)
        layer.mlp.gate_proj.weight.shard_spec = ("model", None)
        layer.mlp.down_proj.weight.shard_spec = (None, "model")

        layer.self_attn.q_proj.weight.shard_spec = ("model", None)
        layer.self_attn.k_proj.weight.shard_spec = ("model", None)
        layer.self_attn.v_proj.weight.shard_spec = ("model", None)
        layer.self_attn.o_proj.weight.shard_spec = (None, "model")

    # for i, (key, value) in enumerate(zip(static_cache.key_cache, static_cache.value_cache)):
    #     static_cache.key_cache[i] = key.to("xla")
    #     static_cache.value_cache[i] = value.to("xla")
    #     xs.mark_sharding(static_cache.key_cache[i], mesh, (None, "model", None, None))
    #     xs.mark_sharding(static_cache.value_cache[i], mesh, (None, "model", None, None))

    # for layer in llama.layers:
    #     xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
    #     xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
    #     xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

    #     xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
    #     xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
    #     xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
    #     xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

    out = llama(input_ids=input_ids, attention_mask=None, past_key_values=static_cache, cache_position=cache_position, use_cache=True)
    out = out.last_hidden_state.cpu().float()
    pcc = calculate_pcc(out, out_cpu.last_hidden_state)
    print(f"LLAMA PCC: {pcc}")
    assert pcc > 0.95