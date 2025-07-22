# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
import torch_xla.core.xla_model as xm

from tt_torch.tools.utils import calculate_pcc


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
    ("ministral3b", "ministral/Ministral-3b-instruct"),
]


@pytest.mark.parametrize(
    "model_info",
    model_info_list,
    ids=[model_info[0] for model_info in model_info_list],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.TORCH, None],
    ids=["op_by_op_torch", "full"],
)
def test_mistral(record_property, model_info, op_by_op):
    __, model_name = model_info
    model_group = "red"

    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    cc.enable_consteval = True

    # TODO Enable PCC/ATOL/Checking - https://github.com/tenstorrent/tt-torch/issues/689
    tester = ThisTester(
        model_name,
        "eval",
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=False,
        assert_pcc=False,
        model_group=model_group,
        backend="tt-experimental",
    )
    tester.test_model()
    tester.finalize()


def test_mistral3b_eager():
    tokenizer = AutoTokenizer.from_pretrained(
        "ministral/Ministral-3b-instruct",
        padding_side="left",
        torch_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "ministral/Ministral-3b-instruct", torch_dtype=torch.bfloat16
    ).eval()

    test_input = "How often does the letter r occur in Mistral?"
    inputs = tokenizer.encode_plus(test_input, return_tensors="pt")

    cpu_outputs = model(**inputs).logits

    device = xm.xla_device()

    model = model.to(device)
    inputs = inputs.to(device)

    tt_outputs = model(**inputs).logits.to("cpu")

    pcc = calculate_pcc(tt_outputs, cpu_outputs)
    print(f"PCC: {pcc}")
