# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Phi-3-mini-128k-instruct: https://huggingface.co/microsoft/Phi-3-mini-128k-instruct
# Phi-3-mini-4k-instruct: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
# Phi-3.5-MoE-instruct: https://huggingface.co/microsoft/Phi-3.5-MoE-instruct

import torch
import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16       
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        return model.generate

    def _load_inputs(self):
        messages = [ 
            {"role": "system", "content": "You are a helpful AI assistant."}, 
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}, 
            {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."}, 
            {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"}, 
        ] 
        self.test_input = messages
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True, 
            return_tensors="pt",
            return_attention_mask=True
        )
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name", ["microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3.5-MoE"]
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_phi(record_property, model_name, mode, op_by_op):
    # same pytest command for reference: pytest tests/models/phi/test_phi_1_1p5_2.py::test_phi[full-microsoft/Phi-3-mini-128k-instruct-eval]
    model_group = "red"
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
        is_token_output=True,
        model_group=model_group,
    )

    results = tester.test_model(assert_eval_token_mismatch=False)

    if mode == "eval":
        decoded_output = tester.tokenizer.decode(results[0])
        print(
            f"""
        model_name: {model_name}
        input: {tester.test_input_messages}
        output: {decoded_output}
        """
        )
    tester.finalize()
