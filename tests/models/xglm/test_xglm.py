# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/facebook/xglm-564M

import torch
import torch.nn.functional as F

from transformers import XGLMTokenizer, XGLMForCausalLM
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
        model = XGLMForCausalLM.from_pretrained(
            "facebook/xglm-564M", torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):
        inputs = self.tokenizer(
            "I wanted to conserve energy.\nI swept the floor in the unoccupied room.",
            return_tensors="pt",
        )
        inputs["labels"] = inputs["input_ids"]
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
def test_xglm(record_property, mode, op_by_op):
    model_name = "XGLM"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    else:
        cc.compile_depth = CompileDepth.TTNN_IR

    tester = ThisTester(
        model_name,
        mode,
        relative_atol=0.02,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    tester.test_model()
