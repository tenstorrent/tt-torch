# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pytest
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, ModelMetadata


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


DEEPSEEK_VARIANTS = [
    ModelMetadata(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        required_atol=0.5,
        model_group="red",
    )
]


@pytest.mark.parametrize("model_info", DEEPSEEK_VARIANTS, ids=lambda x: x.model_name)
@pytest.mark.parametrize(
    "mode",
    ["eval", "train"],
)
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
def test_deepseek_qwen(record_property, model_info, mode, execute_mode):
    if mode == "train":
        pytest.skip()

    cc = CompilerConfig()
    cc.op_by_op_backend = model_info.op_by_op_backend
    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
        cc.compile_depth = execute_mode
    else:
        cc.compile_depth = model_info.compile_depth

    skip_full_eval_test(
        record_property,
        cc,
        model_info.model_name,
        bringup_status="FAILED_RUNTIME",
        reason="Model is too large to fit on single device during execution.",
        model_group=model_info.model_group,
    )

    tester = ThisTester(
        model_name=model_info.model_name,
        model_info=model_info,
        mode=mode,
        compiler_config=cc,
        record_property_handle=record_property,
        required_atol=model_info.required_atol,
        model_group=model_info.model_group,
    )

    tester.test_model()
    tester.finalize()
