# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from PIL import Image
import torch
import pytest
import requests
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from tests.utils import ModelTester


class ThisTester(ModelTester):
    def _load_model(self):
        model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0", torch_dtype=torch.float32, trust_remote_code=True
        )
        torch.set_float32_matmul_precision(["high", "highest"][0])
        image_size = (1024, 1024)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return model

    def _load_inputs(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw)
        inputs = self.transform_image(self.image).unsqueeze(0).to(dtype=torch.float32)
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
def test_RMBG(record_property, mode):
    model_name = "RMBG"
    record_property("model_name", model_name)
    record_property("mode", mode)
    tester = ThisTester(model_name, mode)

    with torch.no_grad():
        results = tester.test_model()
    if mode == "eval":
        predictions = results[-1].sigmoid()
        pred = predictions[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(tester.image.size)
        tester.image.putalpha(mask)
        tester.image.save("no_bg_image.png")