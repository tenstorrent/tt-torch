# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from tests.models.centernet.src.model import Model
from PIL import Image
import torch
import numpy as np
import pytest
from torchvision import transforms
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        """
        The model is from https://github.com/xingyizhou/CenterNet
        """
        # Model
        model = Model()
        model.load_state_dict(torch.load("ctdet_coco_resdcn18.pth", map_location="cpu"))
        return model

    def _load_inputs(self):
        # Images
        image = (
            Image.open("tests/models/centernet/image.png")
            .convert("RGB")
            .resize((512, 512))
        )
        m, s = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=m, std=s),
            ]
        )
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch

    def _extract_outputs(self, output_object):
        return (output_object["hm"], output_object["wh"], output_object["reg"])


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.TORCH, None],
    ids=["op_by_op_torch", "full"],
)
def test_centernet(record_property, mode, op_by_op):
    model_name = "CENTERNET"
    model_group = "red"

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
        model_group=model_group,
    )
    results = tester.test_model()
    tester.finalize()
