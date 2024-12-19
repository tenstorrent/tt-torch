# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/en/model_doc/opt

import torch
from transformers import OPTForCausalLM, GPT2Tokenizer, GenerationConfig
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth


class ThisTester(ModelTester):
    def _load_model(self):
        model = OPTForCausalLM.from_pretrained(
            "facebook/opt-350m", torch_dtype=torch.bfloat16
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
        return model.generate

    def _load_inputs(self):
        prompt = "A chat between a curious"

        model_inputs = self.tokenizer([prompt], return_tensors="pt")

        generation_config = GenerationConfig(max_length=30, do_sample=False)

        model_inputs["generation_config"] = generation_config
        return model_inputs

    def set_model_eval(self, model):
        return model


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.xfail(reason="Need to debug")
def test_opt(record_property, mode, nightly):
    model_name = "OPT"
    record_property("model_name", model_name)
    record_property("mode", mode)

    cc = CompilerConfig()
    if nightly:
        cc.compile_depth = CompileDepth.COMPILE_OP_BY_OP
    tester = ThisTester(model_name, mode, compiler_config=cc)
    results = tester.test_model()
    if mode == "eval":
        tester.tokenizer.batch_decode(results)[0]

    record_property("torch_ttnn", (tester, results))
