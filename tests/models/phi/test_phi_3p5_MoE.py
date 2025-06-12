# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Phi-3.5-MoE-instruct: https://huggingface.co/microsoft/Phi-3.5-MoE-instruct

import torch
import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            return_dict=True,
            torch_dtype=torch.bfloat16,
        )
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return pipe

    def _load_inputs(self):
        self.prompt = """
        Write a short story about a cat:
        """
        arguments = {
            "text_inputs": self.prompt,
            "max_new_tokens": 120,
            "do_sample": True,
        }
        return arguments


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "microsoft/Phi-3.5-MoE-instruct",
    ],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_phi(record_property, model_name, mode, op_by_op):
    model_group = "red"
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_FE_COMPILATION",
        reason="failed to legalize operation 'stablehlo.reduce'",
    )

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
    )

    results = tester.test_model()

    if mode == "eval":
        decoded_output = results[0]["generated_text"]
        print(
            f"""
        model_name: {model_name}
        input: {tester.prompt}
        output: {decoded_output}
        """
        )
    tester.finalize()
