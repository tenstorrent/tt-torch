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
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend

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
        input_batch = torch.stack(
            [transforms(img)] * 4
        )  # stack single image into batch of 4
        input_batch = input_batch.to(torch.bfloat16)
        return input_batch


# Separate lists for models and modes
model_list = [
    # "tf_efficientnet_lite0.in1k",
    # "tf_efficientnet_lite1.in1k",
    # "tf_efficientnet_lite2.in1k",
    # "tf_efficientnet_lite3.in1k",
    "tf_efficientnet_lite4.in1k",
    "ghostnet_100.in1k",
    # "ghostnetv2_100.in1k",
    # "inception_v4.tf_in1k",
    # "mixer_b16_224.goog_in21k",
    # "mobilenetv1_100.ra4_e3600_r224_in1k",
    # "ese_vovnet19b_dw.ra_in1k",
    # "xception71.tf_in1k",
    "dla34.in1k",
    "hrnet_w18.ms_aug_in1k",
]


@pytest.mark.usefixtures("manage_dependencies")
@pytest.mark.parametrize("model_name", model_list)
@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_timm_image_classification(record_property, model_name, mode, op_by_op):
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

    assert_pcc = (
        True
        if model_name
        in [
            "mobilenetv1_100.ra4_e3600_r224_in1k",
            "dla34.in1k",
            "ghostnet_100.in1k",
            "hrnet_w18.ms_aug_in1k",
            "tf_efficientnet_lite0.in1k",
            "tf_efficientnet_lite1.in1k",
            "tf_efficientnet_lite2.in1k",
            "tf_efficientnet_lite3.in1k",
            "tf_efficientnet_lite4.in1k",
            "xception71.tf_in1k",
            "inception_v4.tf_in1k",
        ]
        else False
    )

    required_pcc = (
        0.98
        if model_name
        in [
            "mobilenetv1_100.ra4_e3600_r224_in1k",
        ]
        else 0.99
    )

    # FIXME fails with tt-experimental - https://github.com/tenstorrent/tt-torch/issues/1105
    backend = (
        "tt" if model_name in ["tf_efficientnet_lite4.in1k"] else "tt-experimental"
    )

    model_group = "generality"
    tester = ThisTester(
        model_name,
        mode,
        required_pcc=required_pcc,
        compiler_config=cc,
        assert_pcc=assert_pcc,
        assert_atol=False,
        record_property_handle=record_property,
        model_group=model_group,
        backend=backend,
    )
    results = tester.test_model()

    if mode == "eval":
        top5_probabilities, top5_class_indices = torch.topk(
            results.softmax(dim=1) * 100, k=5
        )
        print(
            f"Model: {model_name} | Predicted class ID: {top5_class_indices[0]} | Probability: {top5_probabilities[0]}"
        )

    tester.finalize()
