# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaConfig


class LlamaTester(ModelTester):
    def _load_model(self):
        model_name = "meta-llama/Llama-3.2-3B"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        if self.model_name == "llamaBasic":
            model.generation_config.cache_implementation = "static"
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
        args = {"input_ids": inputs.input_ids, "use_cache": True}

        if self.model_name == "llamaModel":
            past_key_values = StaticCache(
                config=self.framework_model.config,
                max_batch_size=1,
                max_cache_len=32,
                dtype=self.framework_model.dtype,
            )
            args["past_key_values"] = past_key_values

        return args

    def set_model_eval(self, model):
        return model


class AttnTester(ModelTester):
    def _load_model(self):
        config = LlamaConfig()
        model = LlamaSdpaAttention(config=config, layer_idx=0)
        return model

    def _load_inputs(self):
        batch_size = 1
        seq_length = 32
        hidden_size = self.framework_model.config.hidden_size

        hidden_states = torch.randn(
            batch_size, seq_length, hidden_size, dtype=torch.float32
        )
        attention_mask = torch.ones(
            batch_size, 1, seq_length, seq_length, dtype=torch.float32
        )
        position_ids = torch.arange(seq_length).unsqueeze(0)
        max_generated_length = seq_length
        past_key_values = StaticCache(
            config=self.framework_model.config,
            max_batch_size=batch_size,
            max_cache_len=max_generated_length,
            dtype=hidden_states.dtype,
        )

        args = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_value": past_key_values,
            "use_cache": True,
        }
        return args

    def set_model_eval(self, model):
        return model


model_info_list = [
    ("llamaBasic", "meta-llama/Llama-3.2-3B"),
    ("llamaModel", "meta-llama/Llama-3.2-3B"),
    ("llamaAttn", "meta-llama/Llama-3.2-3B"),
]


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_info",
    model_info_list,
    ids=[model_info[0] for model_info in model_info_list],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_llama_attn(record_property, model_info, mode, op_by_op):
    model_name, _ = model_info
    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO
    if model_name == "llamaModel" or model_name == "llamaBasic":
        tester = LlamaTester(
            model_name,
            mode,
            compiler_config=cc,
            assert_atol=False,
            assert_pcc=False,
            record_property_handle=record_property,
        )
    elif model_name == "llamaAttn":
        tester = AttnTester(
            model_name,
            mode,
            compiler_config=cc,
            assert_atol=False,
            assert_pcc=False,
            record_property_handle=record_property,
        )
    results = tester.test_model()
    tester.finalize()
    return results
