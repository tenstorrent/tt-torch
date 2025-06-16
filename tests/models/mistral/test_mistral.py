# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, ModelMetadata


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", torch_dtype=torch.bfloat16
        )
        m = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        return m

    def _load_inputs(self):
        # Set up sample input
        self.test_input = "How often does the letter r occur in Mistral?"
        inputs = self.tokenizer.encode_plus(self.test_input, return_tensors="pt")
        return inputs


MISTRAL_VARIANTS = [
    ModelMetadata(model_name="mistralai/Mistral-7B-v0.1", model_group="red", compile_depth=CompileDepth.TTNN_IR),
    ModelMetadata(model_name="mistralai/Ministral-8B-Instruct-2410", model_group="red", compile_depth=CompileDepth.TTNN_IR),
    ModelMetadata(model_name="ministral/Ministral-3b-instruct", model_group="red", compile_depth=CompileDepth.TTNN_IR),
]


@pytest.mark.parametrize("model_info", MISTRAL_VARIANTS, ids=lambda x: x.model_name)
@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
def test_mistral(record_property, model_info, mode, execute_mode):
    if model_info.model_name == "ministral/Ministral-3b-instruct":
        pytest.skip(
            " Skipping Mistral-3B model test due to: https://github.com/tenstorrent/tt-torch/issues/905"
        )

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
        model_name_filter=[
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Ministral-8B-Instruct-2410",
        ],
    )

    # TODO Enable PCC/ATOL/Checking - https://github.com/tenstorrent/tt-torch/issues/689
    tester = ThisTester(
        model_name=model_info.model_name,
        model_info=model_info,
        mode=mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=model_info.assert_atol,
        assert_pcc=model_info.assert_pcc,
        model_group=model_info.model_group,
    )
    results = tester.test_model()
    tester.finalize()
