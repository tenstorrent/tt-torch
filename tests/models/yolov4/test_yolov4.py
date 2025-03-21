# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import cv2
import numpy as np
import urllib.request
import pytest

from tests.models.yolov4.src.yolov4 import Yolov4
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


class ThisTester(ModelTester):
    def _load_model(self):
        model = Yolov4()
        return model.to(torch.bfloat16)  # .to(torch.bfloat16)

    def _load_inputs(self):

        img = url_to_image("http://images.cocodataset.org/val2017/000000039769.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (640, 480))  # Resize to model input size
        img = img / 255.0  # Normalize to [0,1]
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW format
        img = [torch.from_numpy(img).float().unsqueeze(0)]  # Add batch dimension
        batch_tensor = torch.cat(img, dim=0).to(torch.bfloat16)
        return batch_tensor


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_yolov4(record_property, mode, op_by_op):
    model_name = "YOLOv4"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name, mode, compiler_config=cc, record_property_handle=record_property
    )
    with torch.no_grad():
        tester.test_model()
