# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Phi 1: https://huggingface.co/microsoft/phi-1
# Phi 1.5: https://huggingface.co/microsoft/phi-1_5
# Phi 2: https://huggingface.co/microsoft/phi-2

import torch
import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        return model

    def _load_inputs(self):
        input_str = '''def print_prime(n):
                        """
                        Print all primes between 1 and n
                        """'''
        self.test_input = input_str
        # inputs = self.tokenizer.encode_plus(
        #     input_str,
        #     return_tensors="pt",
        #     max_length=32,
        #     padding="max_length",
        #     truncation=True,
        # )
        inputs = self.tokenizer.encode_plus(
            input_str,
            return_tensors="pt",
        )
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name", ["microsoft/phi-4"]
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.TORCH],
    ids=["op_by_op_torch"],
)
def test_phi_v4(record_property, model_name, mode, op_by_op):
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
        model_group="red",
    )
    results = tester.test_model()

    if mode == "eval":
        decoded_output = tester.tokenizer.decode(results[0])
        print(
            f"""
        model_name: {model_name}
        input: {tester.test_input}
        output: {decoded_output}
        """
        )
    tester.finalize()
