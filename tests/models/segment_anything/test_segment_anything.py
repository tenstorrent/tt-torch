# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/facebookresearch/segment-anything-2
# Hugging Face version: https://huggingface.co/facebook/sam2-hiera-tiny

import torch
import requests
from PIL import Image
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth


class ThisTester(ModelTester):
    def _load_model(self):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        predictor.set_image(image)
        return predictor

    def _load_inputs(self):
        prompt = "Beautiful thing"
        return prompt

    def run_model(self, model, inputs):
        outputs = model.predict(inputs)
        return outputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.skip(
    reason="Failed to install sam2. sam2 requires Python >=3.10.0 but the default version on Ubuntu 20.04 is 3.8. We found no other pytorch implementation of segment-anything."
)
@pytest.mark.parametrize("op_by_op", [True, False], ids=["op_by_op", "full"])
def test_segment_anything(record_property, mode, op_by_op):
    model_name = "segment-anything"
    record_property("model_name", model_name)
    record_property("mode", mode)

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    tester = ThisTester(model_name, mode, compiler_config=cc)
    results = tester.test_model()

    record_property("torch_ttnn", (tester, results))
