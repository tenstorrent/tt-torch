# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html
# Image URL from: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/mobilenet_ssd/tflite_mobilenet_v2_ssd_1x1.py

import torchvision
import torch
from PIL import Image
from torchvision import transforms
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file


# TODO: RuntimeError: "nms_kernel" not implemented for 'BFloat16'
class ThisTester(ModelTester):
    def _load_model(self):
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        return model  # .to(torch.bfloat16)

    def _load_inputs(self):
        # Image preprocessing
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
        transform = transforms.Compose(
            [transforms.Resize((320, 320)), transforms.ToTensor()]
        )
        img_tensor = [transform(image).unsqueeze(0)]
        batch_tensor = torch.cat(img_tensor, dim=0)
        return batch_tensor  # .to(torch.bfloat16)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_mobilenet_ssd(record_property, mode, op_by_op, data_parallel_mode):
    model_name = "MobileNetSSD"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        # TODO Enable checking - https://github.com/tenstorrent/tt-torch/issues/486
        assert_pcc=False,
        assert_atol=False,
        data_parallel_mode=data_parallel_mode,
    )
    tester.test_model()
    tester.finalize()
