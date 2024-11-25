# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import AutoTokenizer, Qwen2ForTokenClassification
import torch
import pytest
from tests.utils import ModelTester


class ThisTester(ModelTester):
    def _load_model(self):
        return Qwen2ForTokenClassification.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )

    def _load_inputs(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.text = "HuggingFace is a company based in Paris and New York."
        self.inputs = self.tokenizer(
            self.text, add_special_tokens=False, return_tensors="pt"
        )
        return self.inputs


@pytest.mark.parametrize(
    "mode",
    ["eval", "train"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen2-7B",
    ],
)
def test_qwen2_token_classification(record_property, model_name, mode):
    if mode == "train":
        pytest.skip()
    record_property("model_name", model_name)
    record_property("mode", mode)

    tester = ThisTester(model_name, mode)
    with torch.no_grad():
        results = tester.test_model()

    if mode == "eval":
        logits = results.logits
        predicted_token_class_ids = logits.argmax(-1)
        predicted_tokens_classes = [
            tester.model.config.id2label[t.item()] for t in predicted_token_class_ids[0]
        ]
        input_ids = tester.inputs["input_ids"]
        tokens = tester.tokenizer.convert_ids_to_tokens(input_ids[0])
        print(
            f"Model: {model_name} | Tokens: {tokens} | Predictions: {predicted_tokens_classes}"
        )

    record_property("torch_ttnn", (tester, results))
