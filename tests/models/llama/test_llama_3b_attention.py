# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class ThisTester(ModelTester):
    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.attention_layer = self.model.model.layers[0].self_attn
        return self.attention_layer

    def _load_inputs(self):
        self.test_input = "This is a sample text from "
        inputs = self.tokenizer.encode_plus(
            self.test_input,
            return_tensors="pt",
            max_length=32,
            padding="max_length",
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        inputs_embeds = self.model.model.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]

        position_ids = torch.arange(
            0, seq_length, dtype=torch.long, device=inputs_embeds.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        normalized_hidden_states = self.model.model.layers[0].input_layernorm(
            inputs_embeds
        )

        position_embeddings = self.model.model.rotary_emb(inputs_embeds, position_ids)

        causal_mask = self.model.model._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position=torch.arange(seq_length, device=inputs_embeds.device),
            past_key_values=None,
            output_attentions=True,
        )
        arguments = {
            "hidden_states": normalized_hidden_states,
            "attention_mask": causal_mask,
            "position_embeddings": position_embeddings,
            "past_key_value": None,
            "output_attentions": True,
            "use_cache": False,
            "cache_position": torch.arange(seq_length, device=inputs_embeds.device),
        }
        return arguments


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
        assert_pcc=True,
        record_property_handle=record_property,
    )
    results = tester.test_model()
    tester.finalize()
