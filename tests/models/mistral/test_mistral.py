# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


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


model_info_list = [
    ("mistral7b", "mistralai/Mistral-7B-v0.1"),
    ("ministral8b", "mistralai/Ministral-8B-Instruct-2410"),
    ("ministral3b", "ministral/Ministral-3b-instruct"),
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
def test_mistral(record_property, model_info, mode, op_by_op):
    __, model_name = model_info
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
        bringup_status="FAILED_RUNTIME",
        reason="Model is too large to fit on single device during execution.",
        model_group=model_group,
        model_name_filter=[
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Ministral-8B-Instruct-2410",
        ],
    )

    # TODO Enable PCC/ATOL/Checking - https://github.com/tenstorrent/tt-torch/issues/689
    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=False,
        assert_pcc=False,
        model_group=model_group,
    )
    results = tester.test_model()
    tester.finalize()
