# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from third_party.tt_forge_models.yolov4.pytorch import ModelLoader
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file

import cv2
import numpy as np


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        img = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (640, 480))  # Resize to model input size
        img = img / 255.0  # Normalize to [0,1]
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW format
        batch_img = torch.stack([torch.from_numpy(img).float()] * 4)  # Create batch
        batch_tensor = batch_img.to(torch.bfloat16)
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
@pytest.mark.xfail(
    reason="Fails due to error: Could not update shapes based on their tensor sharding attributes, tracked in issue https://github.com/tenstorrent/tt-torch/issues/1215",
    strict=True,
)
def test_yolov4(record_property, mode, op_by_op):
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.automatic_parallelization = True
    cc.mesh_shape = [1, 2]

    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)

    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_info.name,
        mode,
        loader=loader,
        model_info=model_info,
        compiler_config=cc,
        required_pcc=0.98,
        assert_pcc=True,
        assert_atol=False,
        record_property_handle=record_property,
        model_group="red",
        # FIXME fails with tt-experimental - https://github.com/tenstorrent/tt-torch/issues/1105
        backend="tt-legacy",
    )
    with torch.no_grad():
        tester.test_model()
    tester.finalize()
