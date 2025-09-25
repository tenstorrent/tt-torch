# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/deepmind/language-perceiver

from transformers import PerceiverTokenizer, PerceiverForMaskedLM
import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = PerceiverTokenizer.from_pretrained(
            "deepmind/language-perceiver"
        )
        model = PerceiverForMaskedLM.from_pretrained(
            "deepmind/language-perceiver", torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):
        text = "This is an incomplete sentence where some words are missing."
        # prepare input
        encoding = self.tokenizer(text, padding="max_length", return_tensors="pt")
        # mask " missing.". Note that the model performs much better if the masked span starts with a space.
        encoding.input_ids[0, 52:61] = self.tokenizer.mask_token_id
        arguments = {
            "inputs": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
        }
        return arguments


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_perceiver_io(record_property, mode, op_by_op):
    model_name = "Perceiver IO"

    cc = CompilerConfig()
    cc.enable_consteval = True
    if op_by_op:
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
    )
    results = tester.test_model()
    if mode == "eval":
        logits = results.logits
        masked_tokens_predictions = logits[0, 51:61].argmax(dim=-1)
        print(tester.tokenizer.decode(masked_tokens_predictions))

    tester.finalize()
