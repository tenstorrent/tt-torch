# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/en/model_doc/flan-t5

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-small", torch_dtype=torch.bfloat16
        )
        return model.generate

    def _load_inputs(self):
        inputs = self.tokenizer(
            "A step by step recipe to make bolognese pasta:", return_tensors="pt"
        )
        return inputs

    def set_model_eval(self, model):
        return model


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_flan_t5(record_property, mode, op_by_op):
    model_name = "FLAN-T5"

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
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=False,
        assert_atol=False,
        is_token_output=True,
    )
    results = tester.test_model()
    if mode == "eval":
        results = tester.tokenizer.batch_decode(results, skip_special_tokens=True)

    tester.finalize()
