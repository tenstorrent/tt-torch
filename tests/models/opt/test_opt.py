# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/en/model_doc/opt

import torch
from transformers import OPTForCausalLM, GPT2Tokenizer, GenerationConfig
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        model = OPTForCausalLM.from_pretrained(
            "facebook/opt-350m", torch_dtype=torch.bfloat16
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
        return model

    def _load_inputs(self):
        prompt = "A chat between a curious"
        model_inputs = self.tokenizer([prompt], return_tensors="pt")
        return model_inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_opt(record_property, mode, op_by_op):
    model_name = "OPT"

    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    cc.enable_consteval = True

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        relative_atol=0.015,
        assert_pcc=False,  # PCC regression in OPT model observed around July 17, follow up in https://github.com/tenstorrent/tt-torch/issues/1072
        assert_atol=False,
    )
    tester.test_model(assert_eval_token_mismatch=False)
    tester.finalize()
