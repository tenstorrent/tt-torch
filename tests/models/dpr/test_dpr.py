# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/facebook/dpr-reader-single-nq-base

from transformers import DPRReader, DPRReaderTokenizer
import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = DPRReaderTokenizer.from_pretrained(
            "facebook/dpr-reader-single-nq-base"
        )
        model = DPRReader.from_pretrained(
            "facebook/dpr-reader-single-nq-base", torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):
        encoded_inputs = self.tokenizer(
            questions=["What is love ?"],
            titles=["Haddaway"],
            texts=["'What Is Love' is a song recorded by the artist Haddaway"],
            return_tensors="pt",
        )
        return encoded_inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize("op_by_op", [True, False], ids=["op_by_op", "full"])
def test_dpr(record_property, mode, op_by_op):
    model_name = "DPR"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

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
        start_logits = results.start_logits
        end_logits = results.end_logits
        relevance_logits = results.relevance_logits
        print(results)

    tester.finalize()
