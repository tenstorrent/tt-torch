# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/en/model_doc/flan-t5

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-small", torch_dtype=torch.bfloat16
        )
        return model.generate

    def _load_inputs(self):
        inputs = self.tokenizer(
            "A step by step recipe to make bolognese pasta:", return_tensors="pt"
        )
        return inputs

    def set_model_eval(self, model):
        return model


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
def test_flan_t5(record_property, mode, nightly):
    model_name = "FLAN-T5"
    record_property("model_name", model_name)
    record_property("mode", mode)

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if nightly:
        cc.compile_depth = CompileDepth.COMPILE_OP_BY_OP

    tester = ThisTester(model_name, mode, compiler_config=cc)
    results = tester.test_model()
    if mode == "eval":
        results = tester.tokenizer.batch_decode(results, skip_special_tokens=True)

    record_property("torch_ttnn", (tester, results))
