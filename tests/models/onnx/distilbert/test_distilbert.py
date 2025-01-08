# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from onnxruntime import InferenceSession
from onnxruntime import tools
import torch
import pytest
import onnx
from tests.utils import OnnxModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth
import numpy as np


class ThisTester(OnnxModelTester):
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = ORTModelForSequenceClassification.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, subfolder="onnx"
        )
        model = onnx.load(model.model_path)
        return model

    def _load_inputs(self):
        self.text = "Transformers provide state-of-the-art results in NLP."
        inputs = self.tokenizer(self.text, return_tensors="pt")
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name", ["distilbert/distilbert-base-uncased-finetuned-sst-2-english"]
)
def test_distilbert(record_property, model_name, mode, nightly):
    record_property("model_name", model_name)
    record_property("mode", mode)

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if nightly:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    tester = ThisTester(model_name, mode, compiler_config=cc)
    results = tester.test_model()

    if mode == "eval":
        print(f"Model: {model_name} | Input: {tester.text} | Output: {results}")

    record_property("torch_ttnn", (tester, results))
