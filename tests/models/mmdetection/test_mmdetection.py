# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from tests.utils import ModelTester
from mmdet.apis import DetInferencer
from PIL import Image
import requests
import pytest
import torch
import mmengine
import mmdet


# class ThisTester(ModelTester):
#     def _load_model(self):
#         model = DetInferencer(self.model_name)
#         return model

#     def _load_inputs(self):
#         img = "./tests/models/mmdetection/demo.jpg"
#         return img


# @pytest.mark.parametrize(
#     "mode",
#     ["train", "eval"],
# )
# @pytest.mark.parametrize(
#     "model_name",
#     ["rtmdet_tiny_8xb32-300e_coco"],
# )
# def test_mmdetection(record_property, model_name, mode):
#     record_property("model_name", model_name)
#     record_property("mode", mode)

#     tester = ThisTester(model_name, mode)
#     results = tester.test_model()

#     # if mode == "eval":
#     #     print("Results:", results)

#     record_property("torch_ttnn", (tester, results))
