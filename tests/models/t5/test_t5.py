# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        return model.generate

    def _load_inputs(self):
        self.input_text = "translate English to French: How are you?"
        input_ids = self.tokenizer.encode(self.input_text, return_tensors="pt")
        return input_ids

    def set_model_eval(self, model):
        return model


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.xfail(
    reason="Fails due to pt2 compile issue when finishing generation, but we can still generate a graph"
)
@pytest.mark.parametrize("model_name", ["t5-small", "t5-base", "t5-large"])
@pytest.mark.parametrize("op_by_op", [True, False], ids=["op_by_op", "full"])
def test_t5(record_property, model_name, mode, op_by_op):

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    tester = ThisTester(
        model_name, mode, compiler_config=cc, record_property_handle=record_property
    )
    results = tester.test_model()
    if mode == "eval":
        output_text = tester.tokenizer.decode(results[0], skip_special_tokens=True)
        print(
            f"Model: {model_name} | Input: {tester.input_text} | Output: {output_text}"
        )

    tester.finalize()
