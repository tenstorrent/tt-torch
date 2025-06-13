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
        model_dict = dict(model_info_list)
        model_path = model_dict[self.model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            return_dict=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        return self.model

    def _load_inputs(self):
        self.prompt = """
        Write a short story about a cat:
        """
        inputs = self.tokenizer(
            self.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        arguments = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": 120,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        return arguments


model_info_list = [
    ("phi3p5_moe", "microsoft/Phi-3.5-MoE-instruct"),
]


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_info",
    model_info_list,
    ids=[model_info[0] for model_info in model_info_list],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_phi_3p5_moe(record_property, model_info, mode, op_by_op):
    model_group = "red"
    model_name, model_path = model_info

    cc = CompilerConfig()
    cc.enable_consteval = True
    # consteval_parameters is disabled because it results in a memory related crash

    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_FE_COMPILATION",
        model_group=model_group,
        reason="failed to legalize operation 'stablehlo.reduce'",
    )

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
        run_generate=True,
    )

    results = tester.test_model(assert_eval_token_mismatch=False)

    if mode == "eval":
        decoded_output = tester.tokenizer.decode(results[0])
        print(
            f"""
        model_name: {model_name}
        input: {tester.prompt}
        output: {decoded_output}
        """
        )
    tester.finalize()
