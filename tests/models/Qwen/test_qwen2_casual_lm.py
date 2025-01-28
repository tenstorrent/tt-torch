# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/Qwen/Qwen2.5-1.5B

from transformers import AutoTokenizer, Qwen2ForCausalLM, GenerationConfig
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth


class ThisTester(ModelTester):
    def _load_model(self):
        model = Qwen2ForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        return model.generate

    def _load_inputs(self):

        self.text = "Hey, are you conscious? Can you talk to me?"
        input_ids = self.tokenizer(self.text, return_tensors="pt").input_ids
        generation_config = GenerationConfig(max_length=30)
        arguments = {"input_ids": input_ids, "generation_config": generation_config}
        return arguments

    def set_model_eval(self, model):
        return model


@pytest.mark.parametrize(
    "mode",
    ["eval", "train"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen2.5-1.5B",
    ],
)
@pytest.mark.parametrize("op_by_op", [True, False], ids=["op_by_op", "full"])
def test_qwen2_casual_lm(record_property, model_name, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    record_property("model_name", model_name)
    record_property("mode", mode)
    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    tester = ThisTester(model_name, mode, compiler_config=cc)
    results = tester.test_model()

    if mode == "eval":
        gen_text = tester.tokenizer.batch_decode(
            results,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(
            f"Model: {model_name} | Input: {tester.text} | Generated Text: {gen_text}"
        )

    record_property("torch_ttnn", (tester, results))
