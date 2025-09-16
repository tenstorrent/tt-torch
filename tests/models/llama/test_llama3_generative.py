# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from tt_torch.dynamo.backend import BackendOptions
from transformers import AutoTokenizer, AutoModelForCausalLM


class ThisTester(ModelTester):
    def _load_model(self):
        model_name = "meta-llama/Llama-3.2-3B"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            use_cache=True,
        )
        self.model.generation_config.cache_implementation = "static"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.model

    def _load_inputs(self):
        inputs = self.tokenizer.encode_plus(
            "I like taking walks in the",
            return_tensors="pt",
            truncation=True,
            return_attention_mask=True,
        )

        args = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": 32,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        return args


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_llama3_generate(record_property, mode, op_by_op):
    model_name = "meta-llama/Llama-3.2-3B"

    # Setup compilation
    cc = CompilerConfig()

    # Consteval disabled due to 4D Causal Attention Mask evaluation getting constant folded in torchfx
    # due to incorrect tracing of static cache and malformed output missing static cache tensors
    cc.enable_consteval = False
    cc.consteval_parameters = False

    options = BackendOptions()
    options.compiler_config = cc

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=False,
        assert_atol=False,
        run_generate=True,
        backend="tt",
    )

    results = tester.test_model()

    decoded_output = tester.tokenizer.decode(results[0], skip_special_tokens=True)
    print(decoded_output)
