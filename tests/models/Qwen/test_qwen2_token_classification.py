# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import AutoTokenizer, Qwen2ForTokenClassification
import torch
import pytest
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, ModelMetadata


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


QWEN_VARIANTS = [
    ModelMetadata(
        model_name="Qwen/Qwen2-7B",
        model_group="red",
        assert_pcc=False,
        assert_atol=False,
    )
]


@pytest.mark.parametrize("model_info", QWEN_VARIANTS, ids=lambda x: x.model_name)
@pytest.mark.parametrize(
    "mode",
    ["eval", "train"],
)
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
def test_qwen2_token_classification(record_property, model_info, mode, execute_mode):
    if mode == "train":
        pytest.skip()

    cc = CompilerConfig()
    cc.op_by_op_backend = model_info.op_by_op_backend
    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
        cc.compile_depth = execute_mode
    else:
        cc.compile_depth = model_info.compile_depth

    skip_full_eval_test(
        record_property,
        cc,
        model_info.model_name,
        bringup_status="FAILED_RUNTIME",
        reason="Model is too large to fit on single device during execution.",
        model_group=model_info.model_group,
    )

    tester = ThisTester(
        model_name=model_info.model_name,
        model_info=model_info,
        mode=mode,
        assert_pcc=model_info.assert_pcc,
        assert_atol=model_info.assert_atol,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_info.model_group,
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
            f"Model: {model_info.model_name} | Tokens: {tokens} | Predictions: {predicted_tokens_classes}"
        )

    tester.finalize()
