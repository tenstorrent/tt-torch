# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from torchvision import models, transforms
from PIL import Image
import torch
import requests
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


# TODO: RuntimeError: "nms_kernel" not implemented for 'BFloat16'
class ThisTester(ModelTester):
    # pass model_info instead of model_name
    def __init__(self, model_info, mode, *args, **kwargs):
        # model name in model_info[0]
        self.model_info = model_info
        super().__init__(model_info[0], mode, *args, **kwargs)

    def _load_model(self):
        model_name, weights_name = self.model_info
        self.weights = getattr(models.detection, weights_name).DEFAULT
        model = getattr(models.detection, model_name)(
            weights=self.weights
        )  # .to(torch.bfloat16)
        return model

    def _load_inputs(self):
        preprocess = self.weights.transforms()
        # Load and preprocess the image
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0)  # .to(torch.bfloat16)
        return batch_t


model_info_list = [
    ("ssd300_vgg16", "SSD300_VGG16_Weights"),
    ("ssdlite320_mobilenet_v3_large", "SSDLite320_MobileNet_V3_Large_Weights"),
    ("retinanet_resnet50_fpn", "RetinaNet_ResNet50_FPN_Weights"),
    ("retinanet_resnet50_fpn_v2", "RetinaNet_ResNet50_FPN_V2_Weights"),
]


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_info",
    model_info_list,
    ids=[model_info[0] for model_info in model_info_list],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_torchvision_object_detection(record_property, model_info, mode, op_by_op):
    model_name, _ = model_info

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    # TODO - Enable checking - https://github.com/tenstorrent/tt-torch/issues/525
    if model_name == "ssd300_vgg16" or model_name == "ssdlite320_mobilenet_v3_large":
        assert_pcc = False
    else:
        assert_pcc = True

    tester = ThisTester(
        model_info,
        mode,
        assert_pcc=assert_pcc,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()
    if mode == "eval":
        print(f"Model: {model_name} | Output: {results}")

    tester.finalize()
