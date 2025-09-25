# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np
import torch
import pytest
import requests
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file
from ultralytics import YOLO


class ThisTester(ModelTester):
    def _load_model(self):
        # Reference: https://github.com/THU-MIG/yolov10
        file = get_file("test_files/pytorch/yolov10/yolov_10n.pt")
        model = YOLO(file)
        return model.model

    def _load_inputs(self):
        load_path = get_file("https://media.roboflow.com/notebooks/examples/dog.jpeg")
        image = cv2.imread(load_path)
        # image preprocessing:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_size = 640
        resized = cv2.resize(image_rgb, (input_size, input_size))
        normalized = resized.astype(np.float32) / 255.0
        x = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return x


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_yolov10(record_property, mode, op_by_op):
    model_name = "YOLOv10"
    model_group = "red"

    cc = CompilerConfig()
    cc.enable_consteval = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_RUNTIME",
        reason="moreh_softmax_device_operation.cpp Inputs must be of bfloat16 or bfloat8_b type - https://github.com/tenstorrent/tt-torch/issues/730",
        model_group=model_group,
    )

    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
    )
    results = tester.test_model()
    tester.finalize()
