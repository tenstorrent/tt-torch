# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# From: https://huggingface.co/alibaba-damo/mgp-str-base

from PIL import Image
import requests
import torch
from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
import pytest
from tests.utils import ModelTester


class ThisTester(ModelTester):
    def _load_model(self):
        model = MgpstrForSceneTextRecognition.from_pretrained(
            "alibaba-damo/mgp-str-base", torch_dtype=torch.bfloat16
        )
        self.processor = MgpstrProcessor.from_pretrained(
            "alibaba-damo/mgp-str-base", torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):
        url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"  # generated_text = "ticket"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs


@pytest.mark.skip("https://github.com/tenstorrent/tt-torch/issues/96")
@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
def test_mgp_str_base(record_property, mode):
    model_name = "alibaba-damo/mgp-str-base"
    record_property("model_name", model_name)
    record_property("mode", mode)

    tester = ThisTester(model_name, mode)
    results = tester.test_model()

    if mode == "eval":
        logits = results.logits
        generated_text = tester.processor.batch_decode(logits)["generated_text"]
        print(f"Generated text: '{generated_text}'")
        assert generated_text[0] == "ticket"

    record_property("torch_ttnn", (tester, results))