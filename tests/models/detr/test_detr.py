# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from PIL import Image
import requests
import torch
import numpy as np
from torchvision import transforms
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth


class ThisTester(ModelTester):
    def _load_model(self):
        """
        The model is from https://github.com/facebookresearch/detr
        """
        # Model
        model = torch.hub.load(
            "facebookresearch/detr:main", "detr_resnet50", pretrained=True
        ).to(torch.bfloat16)
        return model

    def _load_inputs(self):
        # Images
        input_image = Image.open("tests/models/detr/zidane.jpg")
        m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=m, std=s),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to(torch.bfloat16)
        return input_batch


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
def test_detr(record_property, mode, nightly):
    model_name = "DETR"
    record_property("model_name", model_name)
    record_property("mode", mode)

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if nightly:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
    else:
        cc.compile_depth = CompileDepth.TTNN_IR

    tester = ThisTester(model_name, mode, compiler_config=cc)
    results = tester.test_model()

    if mode == "eval":
        # Results
        print(results)

    record_property("torch_ttnn", (tester, results))
