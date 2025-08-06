# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        config = AutoConfig.from_pretrained(
            "openai/gpt-oss-20b", trust_remote_code=True
        )
        config.quantization_config["quant_method"] = "none"
        self.model = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        return self.model

    def _load_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        messages = [
            {"role": "user", "content": "Who are you?"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
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
def test_gpt_oss(record_property, mode, op_by_op):
    model_name = "GPT-OSS"
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
        run_generate=False,
    )
    results = tester.test_model()
    print(results)
    tester.finalize()
