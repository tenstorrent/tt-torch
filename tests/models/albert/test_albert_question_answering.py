# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/albert#transformers.AlbertForQuestionAnswering

from transformers import AutoTokenizer, AlbertForQuestionAnswering
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth


class ThisTester(ModelTester):
    def _load_model(self):
        model = AlbertForQuestionAnswering.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.question, self.text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        inputs = self.tokenizer(self.question, self.text, return_tensors="pt")
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize("model_name", ["twmkn9/albert-base-v2-squad2"])
@pytest.mark.parametrize("op_by_op", [True, False], ids=["op_by_op", "full"])
def test_albert_question_answering(record_property, model_name, mode, op_by_op):
    record_property("model_name", model_name)
    record_property("mode", mode)

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    tester = ThisTester(model_name, mode, relative_atol=0.01, compiler_config=cc)
    results = tester.test_model()

    if mode == "eval":
        answer_start_index = results.start_logits.argmax()
        answer_end_index = results.end_logits.argmax()

        predict_answer_tokens = tester.inputs.input_ids[
            0, answer_start_index : answer_end_index + 1
        ]
        answer = tester.tokenizer.decode(
            predict_answer_tokens, skip_special_tokens=True
        )

        print(
            f"Model: {model_name} | Question: {tester.question} | Text: {tester.text} | Answer: {answer}"
        )

    record_property("torch_ttnn", (tester, results))
