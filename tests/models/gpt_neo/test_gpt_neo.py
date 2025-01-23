# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/gpt_neo#overview

from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GenerationConfig
import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth


class ThisTester(ModelTester):
    def _load_model(self):
        model = GPTNeoForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-125M", torch_dtype=torch.bfloat16
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        return model.generate

    def _load_inputs(self):
        prompt = (
            "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
            "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
            "researchers was the fact that the unicorns spoke perfect English."
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        generation_config = GenerationConfig(
            max_length=100, do_sample=True, temperature=0.9
        )
        arguments = {"input_ids": input_ids, "generation_config": generation_config}
        return arguments

    def set_model_eval(self, model):
        return model


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.xfail(
    reason="Fails due to pt2 compile issue when finishing generation, but we can still generate a graph"
)
@pytest.mark.parametrize("nightly", [True, False], ids=["nightly", "push"])
def test_gpt_neo(record_property, mode, nightly):
    model_name = "GPTNeo"
    record_property("model_name", model_name)
    record_property("mode", mode)

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if nightly:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    tester = ThisTester(model_name, mode, compiler_config=cc)
    results = tester.test_model()
    if mode == "eval":
        gen_text = tester.tokenizer.batch_decode(results)[0]

    record_property("torch_ttnn", (tester, results))
