# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )

    def _load_inputs(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        prompt = "Who are you?"
        messages = [{"role": "user", "content": prompt}]
        self.text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        self.inputs = self.tokenizer(self.text, return_tensors="pt")
        return self.inputs


@pytest.mark.parametrize(
    "mode",
    ["eval", "train"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_deepseek_qwen(record_property, model_name, mode, op_by_op):
    if mode == "train":
        pytest.skip()

    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        required_atol=0.5,
    )

    tester.test_model()
