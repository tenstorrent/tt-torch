# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/Qwen/Qwen2.5-1.5B

from transformers import AutoTokenizer, Qwen2ForCausalLM
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth


class ThisTester(ModelTester):
    def _load_model(self):
        return Qwen2ForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )

    def _load_inputs(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.text = "Hey, are you conscious? Can you talk to me?"
        self.inputs = self.tokenizer(self.text, return_tensors="pt")
        return self.inputs

    def generate_text(self, model, inputs):
        with torch.no_grad():
            generate_ids = model.generate(inputs.input_ids, max_length=30)
            output_text = self.tokenizer.decode(
                generate_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        return output_text


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen2.5-1.5B",
    ],
)
def test_qwen(record_property, model_name, mode):
    record_property("model_name", model_name)
    record_property("mode", mode)

    compiler_config = CompilerConfig()
    compiler_config.set_compile_depth(CompileDepth.COMPILE_OP_BY_OP)
    tester = ThisTester(model_name, mode, compiler_config)
    results = tester.test_model()

    if mode == "eval":
        model = tester._load_model()
        inputs = tester._load_inputs()
        generated_text = tester.generate_text(model, inputs)

        print(
            f"Model: {model_name} | Input: {tester.text} | Generated Text: {generated_text}"
        )

        assert generated_text.strip(), f"Generated text is empty for model {model_name}"

    record_property("torch_ttnn", (tester, results))
