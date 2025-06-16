# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, ModelMetadata


class ThisTester(ModelTester):
    def _load_model(self):
        # Download model from cloud
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", torch_dtype=torch.bfloat16
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        m = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
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


LLAMA_7B_VARIANTS = [ModelMetadata(model_name="huggyllama/llama-7b", model_group="red", compile_depth=CompileDepth.TTNN_IR)]


@pytest.mark.parametrize("model_info", LLAMA_7B_VARIANTS, ids=lambda x: x.model_name)
@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
def test_llama_7b(record_property, model_info, mode, execute_mode):
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
        assert_pcc=model_info.assert_pcc,
        assert_atol=model_info.assert_atol,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_info.model_group,
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
        model_name: {model_info.model_name}
        input: {tester.test_input}
        output before: {decoded_output}
        """
        )

    tester.finalize()
