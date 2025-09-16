# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from torchvision import models
from PIL import Image
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file


# TODO: RuntimeError: "nms_kernel" not implemented for 'BFloat16'
class ThisTester(ModelTester):
    # pass model_info_tuple instead of model_name
    def __init__(self, model_info_tuple, mode, *args, **kwargs):
        # model name in model_info_tuple[0]
        self.model_info_tuple = model_info_tuple
        super().__init__(model_info_tuple[0], mode, *args, **kwargs)

    def _load_model(self):
        model_name, weights_name = self.model_info_tuple
        self.weights = getattr(models.detection, weights_name).DEFAULT
        model = getattr(models.detection, model_name)(
            weights=self.weights
        )  # .to(torch.bfloat16)
        return model

    def _load_inputs(self):
        preprocess = self.weights.transforms()
        # Load and preprocess the image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
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
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_torchvision_object_detection(
    request, record_property, model_info, mode, op_by_op, data_parallel_mode
):
    model_name, _ = model_info

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
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
        data_parallel_mode=data_parallel_mode,
    )
    results = tester.test_model()

    def print_result(result):
        print(f"Model: {model_name} | Output: {result}")

    if mode == "eval":
        ModelTester.print_outputs(results, data_parallel_mode, print_result)

    tester.finalize()
