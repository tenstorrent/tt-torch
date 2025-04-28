# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, StaticCache


class ThisTester(ModelTester):
    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        return model.model.layers[0].self_attn

    def _load_inputs(self):
        inputs = {}
    
        cache = StaticCache(
            config=self.framework_model.config,
            max_batch_size=1,
            max_cache_len=32,
            dtype=torch.bfloat16,
        )
        inputs["hidden_states"] = torch.randn(1, 32, 3072, dtype=torch.bfloat16)
        inputs["past_key_value"] = cache
        inputs["position_ids"] = torch.arange(32, dtype=torch.long) + 32
        position_embeddings = torch.rand(1, 32, 128, dtype=torch.bfloat16)
        inputs["position_embeddings"] = (position_embeddings, position_embeddings)
        inputs["use_cache"] = True
        inputs["output_attentions"] = False
        inputs["cache_position"] = torch.arange(32, dtype=torch.long)
        return inputs

    def set_model_eval(self, model):
        return model


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

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        assert_atol=False,
        assert_pcc=False,
        record_property_handle=record_property,
    )
    results = tester.test_model()
    tester.finalize()
