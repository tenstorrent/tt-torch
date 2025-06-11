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


class ThisTester(ModelTester):
    # pass model_info instead of model_name
    def __init__(self, model_info, mode, *args, **kwargs):
        # model name in model_info[0]
        self.model_info = model_info
        super().__init__(model_info[0], mode, *args, **kwargs)

    def _load_model(self):
        model_name, weights_name = self.model_info
        self.weights = getattr(models, weights_name).DEFAULT
        model = models.get_model(model_name, weights=self.weights).to(torch.bfloat16)
        return model

    def _load_inputs(self):
        preprocess = self.weights.transforms()
        # Load and preprocess the image
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        img_t = preprocess(image)
        batch_t = torch.stack([img_t] * 4).to(torch.bfloat16)
        return batch_t


# List of model information tuples (model_name, weights_name)
model_info_list = [
    # ("googlenet", "GoogLeNet_Weights"),
    ("densenet121", "DenseNet121_Weights"),
    ("densenet161", "DenseNet161_Weights"),
    ("densenet169", "DenseNet169_Weights"),
    ("densenet201", "DenseNet201_Weights"),
    ("mobilenet_v2", "MobileNet_V2_Weights"),
    ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
    ("resnet18", "ResNet18_Weights"),
    ("resnet34", "ResNet34_Weights"),
    ("resnet50", "ResNet50_Weights"),
    ("resnet101", "ResNet101_Weights"),
    ("resnet152", "ResNet152_Weights"),
    ("resnext50_32x4d", "ResNeXt50_32X4D_Weights"),
    ("resnext101_32x8d", "ResNeXt101_32X8D_Weights"),
    ("resnext101_64x4d", "ResNeXt101_64X4D_Weights"),
    ("vgg11", "VGG11_Weights"),
    ("vgg11_bn", "VGG11_BN_Weights"),
    ("vgg13", "VGG13_Weights"),
    ("vgg13_bn", "VGG13_BN_Weights"),
    ("vgg16", "VGG16_Weights"),
    ("vgg16_bn", "VGG16_BN_Weights"),
    ("vgg19", "VGG19_Weights"),
    ("vgg19_bn", "VGG19_BN_Weights"),
    ("vit_b_16", "ViT_B_16_Weights"),
    # ("vit_b_32", "ViT_B_32_Weights"),
    # ("vit_l_16", "ViT_L_16_Weights"),
    # ("vit_l_32", "ViT_L_32_Weights"),
    ("vit_h_14", "ViT_H_14_Weights"),
    ("wide_resnet50_2", "Wide_ResNet50_2_Weights"),
    ("wide_resnet101_2", "Wide_ResNet101_2_Weights"),
    ("regnet_y_400mf", "RegNet_Y_400MF_Weights"),
    ("regnet_y_800mf", "RegNet_Y_800MF_Weights"),
    ("regnet_y_1_6gf", "RegNet_Y_1_6GF_Weights"),
    ("regnet_y_3_2gf", "RegNet_Y_3_2GF_Weights"),
    ("regnet_y_8gf", "RegNet_Y_8GF_Weights"),
    ("regnet_y_16gf", "RegNet_Y_16GF_Weights"),
    ("regnet_y_32gf", "RegNet_Y_32GF_Weights"),
    ("regnet_y_128gf", "RegNet_Y_128GF_Weights"),
    ("regnet_x_400mf", "RegNet_X_400MF_Weights"),
    ("regnet_x_800mf", "RegNet_X_800MF_Weights"),
    ("regnet_x_1_6gf", "RegNet_X_1_6GF_Weights"),
    ("regnet_x_3_2gf", "RegNet_X_3_2GF_Weights"),
    ("regnet_x_8gf", "RegNet_X_8GF_Weights"),
    ("regnet_x_16gf", "RegNet_X_16GF_Weights"),
    ("regnet_x_32gf", "RegNet_X_32GF_Weights"),
    # ("swin_t", "Swin_T_Weights"),
    # ("swin_s", "Swin_S_Weights"),
    # ("swin_b", "Swin_B_Weights"),
    # ("swin_v2_t", "Swin_V2_T_Weights"),
    # ("swin_v2_s", "Swin_V2_S_Weights"),
    # ("swin_v2_b", "Swin_V2_B_Weights"),
]


@pytest.mark.parametrize(
    "model_info",
    model_info_list,
    ids=[info[0] for info in model_info_list],
)
@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_torchvision_image_classification(
    request, record_property, model_info, mode, op_by_op
):
    if mode == "train":
        pytest.skip()

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.automatic_parallelization = True
    cc.mesh_shape = [1, 2]
    cc.dump_debug = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    # TODO Enable checking (vit_h_14) - https://github.com/tenstorrent/tt-torch/issues/491
    model_name = model_info[0]
    assert_pcc = False if model_name in ["vit_h_14"] else True
    assert_atol = False

    model_group = "red" if model_name == "swin_v2_s" else "generality"

    # Out of Memory: Not enough space to allocate 336691200 B DRAM buffer across 12 banks, where each bank needs to store 28057600 B
    if model_name == "vit_h_14" and cc.compile_depth == CompileDepth.EXECUTE:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Out of Memory: Not enough space to allocate in DRAM error - https://github.com/tenstorrent/tt-torch/issues/793",
                strict=True,
            )
        )

    tester = ThisTester(
        model_info,
        mode,
        required_pcc=0.96,
        assert_pcc=assert_pcc,
        assert_atol=assert_atol,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
    )
    results = tester.test_model()

    if mode == "eval":
        # Print the top 5 predictions
        _, indices = torch.topk(results, 5)
        print(f"Model: {model_name} | Top 5 predictions: {indices[0].tolist()}")

    tester.finalize()
