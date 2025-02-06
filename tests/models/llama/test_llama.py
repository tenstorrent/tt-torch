# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth


class ThisTester(ModelTester):
    def _load_model(self):
        # Download model from cloud
        model_name = "huggyllama/llama-7b"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", torch_dtype=torch.bfloat16
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        for param in m.parameters():
            param.requires_grad = False
        return m

    def _load_inputs(self):
        # Set up sample input
        self.test_input = "This is a sample text from "
        inputs = self.tokenizer.encode_plus(
            self.test_input,
            return_tensors="pt",
            max_length=32,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
        )
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.xfail(
    reason="llama-7b is too large to fit on single device, but we can still generate a graph"
)
@pytest.mark.parametrize("op_by_op", [True, False], ids=["op_by_op", "full"])
def test_llama(record_property, mode, op_by_op):
    model_name = "Llama"
    record_property("model_name", model_name)
    record_property("mode", mode)

    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    tester = ThisTester(
        model_name, mode, assert_pcc=False, assert_atol=False, compiler_config=cc
    )
    results = tester.test_model()
    if mode == "eval":
        # Helper function to decode output to human-readable text
        def decode_output(outputs):
            next_token_logits = outputs.logits[:, -1]
            next_token = next_token_logits.softmax(dim=-1).argmax()
            return tester.tokenizer.decode([next_token])

        decoded_output = decode_output(results)

        print(
            f"""
        model_name: {model_name}
        input: {tester.test_input}
        output before: {decoded_output}
        """
        )

    record_property("torch_ttnn", (tester, results))
