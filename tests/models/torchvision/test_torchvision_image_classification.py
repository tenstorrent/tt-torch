# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from torchvision import models
from PIL import Image
import torch
import pytest
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    OpByOpBackend,
    ModelMetadata,
)
from third_party.tt_forge_models.tools.utils import get_file


# TorchVision (TV) model metadata
# Special Subclass of ModelMetada made for this test to include weight_names attr
class TVModelMetadata(ModelMetadata):
    def __init__(self, model_name, weights_name, required_pcc=0.96, **kwargs):
        super().__init__(model_name=model_name, required_pcc=required_pcc, **kwargs)
        self.weights_name = weights_name


class ThisTester(ModelTester):
    def _load_model(self):
        self.weights = getattr(models, self.model_info.weights_name).DEFAULT
        model = models.get_model(self.model_name, weights=self.weights).to(
            torch.bfloat16
        )
        return model

    def _load_inputs(self):
        preprocess = self.weights.transforms()
        # Load and preprocess the image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0).to(torch.bfloat16)
        return batch_t


# List of Torch Vision Model Metadatas (TVModelMetadata)
TORCHVISION_MODELS = [
    TVModelMetadata(
        model_name="googlenet",
        weights_name="GoogLeNet_Weights",
        assert_pcc=True,
        compile_depth=CompileDepth.TTNN_IR,
    ),
    TVModelMetadata(
        model_name="densenet121",
        weights_name="DenseNet121_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="densenet161",
        weights_name="DenseNet161_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="densenet169",
        weights_name="DenseNet169_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="densenet201",
        weights_name="DenseNet201_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="mobilenet_v2",
        weights_name="MobileNet_V2_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="mobilenet_v3_small",
        weights_name="MobileNet_V3_Small_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="mobilenet_v3_large",
        weights_name="MobileNet_V3_Large_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="resnet18",
        weights_name="ResNet18_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="resnet34",
        weights_name="ResNet34_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="resnet50",
        weights_name="ResNet50_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="resnet101",
        weights_name="ResNet101_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="resnet152",
        weights_name="ResNet152_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="resnext50_32x4d",
        weights_name="ResNeXt50_32X4D_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="resnext101_32x8d",
        weights_name="ResNeXt101_32X8D_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="resnext101_64x4d",
        weights_name="ResNeXt101_64X4D_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="vgg11",
        weights_name="VGG11_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="vgg11_bn",
        weights_name="VGG11_BN_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="vgg13",
        weights_name="VGG13_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="vgg13_bn",
        weights_name="VGG13_BN_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="vgg16",
        weights_name="VGG16_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="vgg16_bn",
        weights_name="VGG16_BN_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="vgg19",
        weights_name="VGG19_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="vgg19_bn",
        weights_name="VGG19_BN_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="vit_b_16",
        weights_name="ViT_B_16_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="vit_b_32",
        weights_name="ViT_B_32_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="vit_l_16",
        weights_name="ViT_L_16_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="vit_l_32",
        weights_name="ViT_L_32_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="vit_h_14",
        weights_name="ViT_H_14_Weights",
    ),
    TVModelMetadata(
        model_name="wide_resnet50_2",
        weights_name="Wide_ResNet50_2_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="wide_resnet101_2",
        weights_name="Wide_ResNet101_2_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="regnet_y_400mf",
        weights_name="RegNet_Y_400MF_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="regnet_y_800mf",
        weights_name="RegNet_Y_800MF_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="regnet_y_1_6gf",
        weights_name="RegNet_Y_1_6GF_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="regnet_y_3_2gf",
        weights_name="RegNet_Y_3_2GF_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="regnet_y_8gf",
        weights_name="RegNet_Y_8GF_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="regnet_y_16gf",
        weights_name="RegNet_Y_16GF_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="regnet_y_32gf",
        weights_name="RegNet_Y_32GF_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="regnet_y_128gf",
        weights_name="RegNet_Y_128GF_Weights",
        assert_pcc=True,
        compile_depth=CompileDepth.TTNN_IR,
    ),
    TVModelMetadata(
        model_name="regnet_x_400mf",
        weights_name="RegNet_X_400MF_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="regnet_x_800mf",
        weights_name="RegNet_X_800MF_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="regnet_x_1_6gf",
        weights_name="RegNet_X_1_6GF_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="regnet_x_3_2gf",
        weights_name="RegNet_X_3_2GF_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="regnet_x_8gf",
        weights_name="RegNet_X_8GF_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="regnet_x_16gf",
        weights_name="RegNet_X_16GF_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="regnet_x_32gf",
        weights_name="RegNet_X_32GF_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="swin_t",
        weights_name="Swin_T_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="swin_s",
        weights_name="Swin_S_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="swin_b",
        weights_name="Swin_B_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="swin_v2_t",
        weights_name="Swin_V2_T_Weights",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="swin_v2_s",
        weights_name="Swin_V2_S_Weights",
        model_group="red",
        assert_pcc=True,
    ),
    TVModelMetadata(
        model_name="swin_v2_b",
        weights_name="Swin_V2_B_Weights",
        assert_pcc=True,
    ),
]


@pytest.mark.parametrize(
    "model_info",
    TORCHVISION_MODELS,
    ids=lambda x: x.model_name,
)
@pytest.mark.parametrize("mode", ["eval"])
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_torchvision_image_classification(
    record_property,
    model_info,
    mode,
    execute_mode,
    data_parallel_mode,
):
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    # check if OpByOp
    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
        if data_parallel_mode == "data_parallel":
            pytest.skip("Op-by-op not supported in data parallel mode")
        cc.compile_depth = execute_mode
    # applying overrides from model_metadata if EXECUTE
    else:
        cc.compile_depth = model_info.compile_depth
    cc.op_by_op_backend = model_info.op_by_op_backend

    skip_full_eval_test(
        record_property,
        cc,
        model_info.model_name,
        bringup_status="FAILED_RUNTIME",
        reason="Out of Memory: Not enough space to allocate 336691200 B DRAM buffer across 12 banks, where each bank needs to store 28057600 B - https://github.com/tenstorrent/tt-torch/issues/793",
        model_group=model_info.model_group,
        model_name_filter=[
            "vit_h_14",
        ],
    )

    tester = ThisTester(
        model_name=model_info.model_name,
        model_info=model_info,
        mode=mode,
        required_pcc=model_info.required_pcc,
        assert_pcc=model_info.assert_pcc,
        assert_atol=model_info.assert_atol,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_info.model_group,
        data_parallel_mode=data_parallel_mode,
    )
    results = tester.test_model()

    def print_result(result):
        _, indices = torch.topk(result, 5)
        print(
            f"Model: {model_info.model_name} | Top 5 predictions: {indices[0].tolist()}"
        )

    if mode == "eval":
        ModelTester.print_outputs(results, False, print_result)

    tester.finalize()
