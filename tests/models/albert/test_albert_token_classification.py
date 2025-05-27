# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/albert#transformers.AlbertForTokenClassification

from transformers import AutoTokenizer, AlbertForTokenClassification
import torch
import pytest
from tests.utils import ModelTester

from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        return AlbertForTokenClassification.from_pretrained(
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
    ["eval"],
)
@pytest.mark.parametrize("model_name", ["albert/albert-base-v2"])
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_albert_token_classification(
    record_property, model_name, mode, op_by_op, data_parallel_mode
):

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=True,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        model_name_suffix="-token-cls",
        data_parallel_mode=data_parallel_mode,
    )
    results = tester.test_model()

    if mode == "eval":
        if data_parallel_mode:
            for i in range(len(results)):
                logits = results[i].logits
                predicted_token_class_ids = logits.argmax(-1)

                # Note that tokens are classified rather then input words which means that
                # there might be more predicted token classes than words.
                # Multiple token classes might account for the same word
                predicted_tokens_classes = [
                    tester.framework_model.config.id2label[t.item()]
                    for t in predicted_token_class_ids[0]
                ]

                input_ids = tester.inputs["input_ids"]
                tokens = tester.tokenizer.convert_ids_to_tokens(input_ids[0])
                print(
                    f"Model: {model_name} | Device: {i} | Tokens: {tokens} | Predictions: {predicted_tokens_classes}"
                )
        else:
            logits = results.logits
            predicted_token_class_ids = logits.argmax(-1)

            # Note that tokens are classified rather then input words which means that
            # there might be more predicted token classes than words.
            # Multiple token classes might account for the same word
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
