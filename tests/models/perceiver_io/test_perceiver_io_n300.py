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
        texts = [
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
            "This is an incomplete sentence where some words are missing.",
        ]
        # prepare batch input
        encoding = self.tokenizer(texts, padding="max_length", return_tensors="pt")
        # mask " missing." for each batch element (adjust indices as needed)
        for i in range(len(texts)):
            encoding.input_ids[i, 52:61] = self.tokenizer.mask_token_id
        arguments = {
            "inputs": encoding.input_ids,  # shape: (batch_size, seq_len)
            "attention_mask": encoding.attention_mask,  # shape: (batch_size, seq_len)
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
    cc.consteval_parameters = True
    cc.automatic_parallelization = True
    cc.mesh_shape = [1, 2]
    cc.dump_debug = True
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
        batch_size = logits.shape[0]
        for i in range(batch_size):
            masked_tokens_predictions = logits[i, 51:61].argmax(dim=-1)
            print(
                f"Decoded for batch {i}: {tester.tokenizer.decode(masked_tokens_predictions)}"
            )

    tester.finalize()
