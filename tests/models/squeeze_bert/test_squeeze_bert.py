# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/en/model_doc/squeezebert#transformers.SqueezeBertForSequenceClassification

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "squeezebert/squeezebert-mnli", torch_dtype=torch.bfloat16
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "squeezebert/squeezebert-mnli", torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):
        inputs = self.tokenizer("Hello, my dog is cute", return_tensors="pt")
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_squeeze_bert(record_property, mode, op_by_op):
    model_name = "SqueezeBERT"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        # FIXME - Open issue
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=False,
    )
    results = tester.test_model()
    if mode == "eval":
        logits = results.logits
        predicted_class_id = logits.argmax().item()

    tester.finalize()
