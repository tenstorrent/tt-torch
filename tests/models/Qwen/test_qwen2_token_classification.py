# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import AutoTokenizer, Qwen2ForTokenClassification
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


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
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_qwen2_token_classification(record_property, model_name, mode, op_by_op):
    if mode == "train":
        pytest.skip()

    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    if op_by_op is None and cc.compile_depth == CompileDepth.EXECUTE:
        pytest.skip("Model is too large to fit on single device during execution.")

    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    with torch.no_grad():
        results = tester.test_model()

    if mode == "eval":
        logits = results.logits
        predicted_token_class_ids = logits.argmax(-1)
        predicted_tokens_classes = [
            tester.framework_model.config.id2label[t.item()]
            for t in predicted_token_class_ids[0]
        ]
        input_ids = tester.inputs["input_ids"]
        tokens = tester.tokenizer.convert_ids_to_tokens(input_ids[0])
        print(
            f"Model: {model_name} | Tokens: {tokens} | Predictions: {predicted_tokens_classes}"
        )

    tester.finalize()
