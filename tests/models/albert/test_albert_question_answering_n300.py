# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/albert#transformers.AlbertForQuestionAnswering

from transformers import AutoTokenizer, AlbertForQuestionAnswering
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


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
        batch_size = 32
        self.question = ["Who was Jim Henson?"] * batch_size
        self.text = ["Jim Henson was a nice puppet"] * batch_size
        inputs = self.tokenizer(
            self.question, self.text, return_tensors="pt", padding=True
        )
        return inputs
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize("model_name", ["twmkn9/albert-base-v2-squad2"])
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_albert_question_answering(record_property, model_name, mode, op_by_op):

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        required_pcc=0.975,
        relative_atol=0.01,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=True,
        assert_atol=False,
    )
    results = tester.test_model()

    if mode == "eval":
        batch_size = results.start_logits.shape[0]
        for i in range(batch_size):
            answer_start_index = results.start_logits[i].argmax().item()
            answer_end_index = results.end_logits[i].argmax().item()
            predict_answer_tokens = tester.inputs["input_ids"][
                i, answer_start_index : answer_end_index + 1
            ]
            answer = tester.tokenizer.decode(
                predict_answer_tokens, skip_special_tokens=True
            )
            print(
                f"Sample {i}: Model: {model_name} | Question: {tester.question[i]} | Text: {tester.text[i]} | Answer: {answer}"
            )

    tester.finalize()
