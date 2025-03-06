# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/albert#transformers.AlbertForSequenceClassification

from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch
import pytest
from tests.utils import ModelTester

from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        return AlbertForSequenceClassification.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )

    def _load_inputs(self):
        self.tokenizer = AlbertTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.input_text = "Hello, my dog is cute."
        self.inputs = self.tokenizer(self.input_text, return_tensors="pt")
        return self.inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize("model_name", ["textattack/albert-base-v2-imdb"])
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_albert_sequence_classification(record_property, model_name, mode, op_by_op):

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
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()

    if mode == "eval":
        logits = results.logits
        predicted_class_id = logits.argmax().item()
        predicted_label = tester.framework_model.config.id2label[predicted_class_id]

        print(
            f"Model: {model_name} | Input: {tester.input_text} | Label: {predicted_label}"
        )

    tester.finalize()
