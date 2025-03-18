# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# https://huggingface.co/microsoft/phi-1

import torch
import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend

class ThisTester(ModelTester):
    def _load_model(self):
        model_name = "microsoft/phi-1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return AutoModelForCausalLM.from_pretrained(model_name)
    
    def _load_inputs(self):
        input_str = '''def print_prime(n):
                        """
                        Print all primes between 1 and n
                        """'''
        self.test_input = input_str
        inputs = self.tokenizer(input_str, return_tensors="pt", return_attention_mask=False)
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

def test_phi_1(record_property, mode, op_by_op):
    model_name = "Phi 1"

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
        relative_atol=0.01,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()
    
    if mode == "eval":
        def decode_output(outputs):
            next_token_logits = outputs.logits[:, -1]
            next_token = next_token_logits.softmax(dim=-1).argmax()
            return tester.tokenizer.decode([next_token])
        decoded_output = decode_output(results)
        print(
            f"""
        model_name: {model_name}
        input: {tester.test_input}
        output: {decoded_output}
        """
        )
    tester.finalize()
    