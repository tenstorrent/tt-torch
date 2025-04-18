# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/yolo_v3/pytorch_yolov3_holli.py

import torch
from PIL import Image
from torchvision import transforms
import requests
import os
from pathlib import Path
import pytest

from tests.models.yolov3.holli_src.yolov3 import *
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        # Download model weights
        url = "https://www.ollihuotari.com/data/yolov3_pytorch/yolov3_coco_01.h5"

        if (
            "DOCKER_CACHE_ROOT" in os.environ
            and Path(os.environ["DOCKER_CACHE_ROOT"]).exists()
        ):
            download_dir = Path(os.environ["DOCKER_CACHE_ROOT"]) / "custom_weights"
        else:
            download_dir = Path.home() / ".cache/custom_weights"
        download_dir.mkdir(parents=True, exist_ok=True)

        load_path = download_dir / url.split("/")[-1]
        if not load_path.exists():
            response = requests.get(url, stream=True)
            with open(str(load_path), "wb") as f:
                f.write(response.content)

        # Load model
        model = Yolov3(num_classes=80)
        model.load_state_dict(
            torch.load(
                str(load_path),
                map_location=torch.device("cpu"),
            )
        )
        return model.to(torch.bfloat16)

    def _load_inputs(self):
        # Image preprocessing
        image_url = (
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
        )
        image = Image.open(requests.get(image_url, stream=True).raw)
        transform = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor()]
        )
        img_tensor = [transform(image).unsqueeze(0)]
        batch_tensor = torch.cat(img_tensor, dim=0).to(torch.bfloat16)
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
def test_yolov3(record_property, mode, op_by_op):
    model_name = "YOLOv3"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    tester.test_model()
    tester.finalize()
