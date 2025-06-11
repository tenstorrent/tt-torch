# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from PIL import Image
import torch
import onnx
import os
import numpy as np
from torchvision import transforms
import pytest
from tests.utils import OnnxModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file


class ThisTester(OnnxModelTester):
    def _load_model(self):
        """
        The model is from https://github.com/facebookresearch/detr
        """
        # Model
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.torch_model = torch.hub.load(
            "facebookresearch/detr:main", "detr_resnet50", pretrained=True
        )
        model = self.torch_model.eval()

        # Export to ONNX
        torch.onnx.export(model, self._load_torch_inputs(), f"{self.model_name}.onnx")
        model = onnx.load(f"{self.model_name}.onnx")
        os.remove(f"{self.model_name}.onnx")
        return model

    def _load_torch_inputs(self):
        # Images
        # Local cache of https://user-images.githubusercontent.com/87515266/177115824-289876a8-7d2d-45a8-9fa6-ab4f37b940e4.jpg (zidane)
        image_file = get_file("test_images/zidane_1280x720.jpg")
        input_image = Image.open(str(image_file))
        m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=m, std=s),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch

    def _extract_outputs(self, output_object):
        return (output_object["pred_logits"], output_object["pred_boxes"])


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, None],
    ids=["op_by_op_stablehlo", "full"],
)
def test_detr_onnx(record_property, mode, op_by_op):
    model_name = "DETR_onnx"
    model_group = "red"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    if op_by_op is not None:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        cc.op_by_op_backend = op_by_op

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_RUNTIME",
        reason="Out of Memory: Not enough space to allocate 59244544 B L1 buffer across 64 banks, where each bank needs to store 925696 B - https://github.com/tenstorrent/tt-torch/issues/729",
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

    if mode == "eval":
        # Results
        print(results)

    tester.finalize()
