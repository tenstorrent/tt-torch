# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch_xla.core.xla_model as xm

from tt_torch.tools.utils import (
    calculate_pcc,
)


class ThisTester(ModelTester):
    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return model

    def _load_inputs(self):
        self.test_input = "This is a sample text from "
        inputs = self.tokenizer.encode_plus(
            self.test_input,
            return_tensors="pt",
            max_length=32,
            padding="max_length",
            truncation=True,
        )
        return inputs


@pytest.mark.parametrize("model_name", ["meta-llama/Llama-3.2-3B"])
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.TORCH, None],
    ids=["op_by_op_torch", "full"],
)
def test_llama_3b(record_property, model_name, op_by_op):
    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    cc.enable_consteval = True

    tester = ThisTester(
        model_name,
        "eval",
        compiler_config=cc,
        assert_atol=False,
        assert_pcc=True,
        required_pcc=0.96,
        record_property_handle=record_property,
        backend="tt-experimental",
    )
    tester.test_model()
    tester.finalize()


def test_llama_3b_eager():
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B", torch_dtype=torch.bfloat16
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B", torch_dtype=torch.bfloat16
    )
    tokenizer.pad_token = tokenizer.eos_token
    test_input = "This is a sample text from "
    inputs = tokenizer.encode_plus(
        test_input,
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        truncation=True,
    )
    cpu_outputs = model(**inputs).logits

    device = xm.xla_device()
    model = model.to(device)
    inputs = inputs.to(device)

    tt_outputs = model(**inputs).logits.to("cpu")

    pcc = calculate_pcc(tt_outputs, cpu_outputs)
    print(f"PCC: {pcc}")
    assert pcc >= 0.96
