# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/facebook/xglm-564M

import torch
import torch.nn.functional as F

from transformers import XGLMTokenizer, XGLMForCausalLM
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth


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
@pytest.mark.parametrize("nightly", [True, False], ids=["nightly", "push"])
def test_xglm(record_property, mode, nightly):
    model_name = "XGLM"
    record_property("model_name", model_name)
    record_property("mode", mode)

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if nightly:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
    else:
        cc.compile_depth = CompileDepth.TTNN_IR

    tester = ThisTester(model_name, mode, relative_atol=0.02, compiler_config=cc)
    results = tester.test_model()

    record_property("torch_ttnn", (tester, results))
