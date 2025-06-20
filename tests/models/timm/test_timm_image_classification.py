# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/timm
# PyTorch Image Models (timm) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

from urllib.request import urlopen
from PIL import Image
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, ModelMetadata

dependencies = ["timm==1.0.9"]


class ThisTester(ModelTester):
    def _load_model(self):
        import timm

        model = timm.create_model(self.model_name, pretrained=True)
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        import timm

        img = Image.open(
            urlopen(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
            )
        )
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.framework_model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        input_batch = transforms(img).unsqueeze(
            0
        )  # unsqueeze single image into batch of 1
        input_batch = input_batch.to(torch.bfloat16)
        return input_batch


# Separate lists for models and modes
model_list = [
    ModelMetadata(
        model_name="tf_efficientnet_lite0.in1k",
        assert_pcc=True,
        required_pcc=0.98,
    ),
    ModelMetadata(
        model_name="tf_efficientnet_lite1.in1k",
        assert_pcc=True,
        required_pcc=0.98,
    ),
    ModelMetadata(
        model_name="tf_efficientnet_lite2.in1k",
        assert_pcc=True,
        required_pcc=0.98,
    ),
    ModelMetadata(
        model_name="tf_efficientnet_lite3.in1k",
        assert_pcc=True,
        required_pcc=0.98,
    ),
    ModelMetadata(
        model_name="tf_efficientnet_lite4.in1k",
        assert_pcc=True,
    ),
    ModelMetadata(
        model_name="ghostnet_100.in1k",
        assert_pcc=True,
    ),
    ModelMetadata(
        model_name="ghostnetv2_100.in1k",
        model_group="generality",
        compile_depth=CompileDepth.TTNN_IR,
    ),
    ModelMetadata(
        model_name="inception_v4.tf_in1k",
        assert_pcc=True,
        required_pcc=0.98,
    ),
    ModelMetadata(
        model_name="mixer_b16_224.goog_in21k",
    ),
    ModelMetadata(
        model_name="mobilenetv1_100.ra4_e3600_r224_in1k",
        assert_pcc=True,
        required_pcc=0.95,
    ),
    ModelMetadata(
        model_name="ese_vovnet19b_dw.ra_in1k",
    ),
    ModelMetadata(
        model_name="xception71.tf_in1k",
        assert_pcc=True,
        required_pcc=0.98,
    ),
    ModelMetadata(
        model_name="dla34.in1k",
        assert_pcc=True,
    ),
    ModelMetadata(
        model_name="hrnet_w18.ms_aug_in1k",
        assert_pcc=True,
    ),
]


@pytest.mark.usefixtures("manage_dependencies")
@pytest.mark.parametrize("model_info", model_list, ids=lambda x: x.model_name)
@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode",
    [False, True],
    ids=["single_device", "data_parallel"],
)
def test_timm_image_classification(
    record_property, model_info, mode, execute_mode, data_parallel_mode
):
    if mode == "train":
        pytest.skip()

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.op_by_op_backend = model_info.op_by_op_backend
    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
        cc.compile_depth = execute_mode
    else:
        cc.compile_depth = model_info.compile_depth

    tester = ThisTester(
        model_name=model_info.model_name,
        model_info=model_info,
        mode=mode,
        required_pcc=model_info.required_pcc,
        compiler_config=cc,
        assert_pcc=model_info.assert_pcc,
        assert_atol=model_info.assert_atol,
        record_property_handle=record_property,
        model_group=model_info.model_group,
        data_parallel_mode=data_parallel_mode,
    )
    results = tester.test_model()

    def print_result(result):
        top5_probabilities, top5_class_indices = torch.topk(
            result.softmax(dim=1) * 100, k=5
        )
        print(
            f"Model: {model_info.model_name} | Predicted class ID: {top5_class_indices[0]} | Probability: {top5_probabilities[0]}"
        )

    if mode == "eval":
        ModelTester.print_outputs(results, data_parallel_mode, print_result)

    tester.finalize()
