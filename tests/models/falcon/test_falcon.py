# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend, ModelMetadata


class ThisTester(ModelTester):
    def _load_model(self):
        # Download model from cloud
        model_name = "tiiuae/falcon-7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", torch_dtype=torch.bfloat16
        )
        m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        return m

    def _load_inputs(self):
        # Set up sample input
        self.test_input = "This is a sample text from "
        inputs = self.tokenizer(self.test_input, return_tensors="pt")
        return inputs


# metadata for Falcon model
FALCON_VARIANT = [
    ModelMetadata(model_name="falcon-7b-instruct", expected_compile_depth=CompileDepth.TTNN_IR, 
                  expected_op_by_op_backend=OpByOpBackend.STABLEHLO,)
]

@pytest.mark.parametrize("model_info", FALCON_VARIANT, ids=lambda x: x.model_name)
@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
     "execute_mode",
     [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
     ids=["op_by_op","full"],
)
def test_falcon(record_property, mode, execute_mode, model_metadata_fixture):
    model_name = "Falcon"

    cc = CompilerConfig
    cc.enable_consteval = True
    cc.consteval_parameters = True

    # Line below is used to get metadata from dict in the case of testEfficientNet.py
    model_metadata = model_metadata_fixture

    # set default compiler config
    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
        cc.compile_depth = execute_mode

    # applying overrides from model_metadata if it exists
    if model_metadata:
        if model_metadata.compile_depth is not None:
            cc.compile_depth = model_metadata.compile_depth
        if model_metadata.op_by_op_backend is not None:
            cc.op_by_op_backend = model_metadata.op_by_op_backend


    tester = ThisTester(
        model_name,
        mode,
        relative_atol=0.015,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=False,
        assert_atol=False,
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

    tester.finalize()
