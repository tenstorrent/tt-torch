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
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    ModelMetadata,
    OpByOpBackend,
)
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
        image_file = get_file(
            "https://huggingface.co/spaces/nakamura196/yolov5-char/resolve/8a166e0aa4c9f62a364dafa7df63f2a33cbb3069/ultralytics/yolov5/data/images/zidane.jpg"
        )
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


DETR_ONNX_VARIANTS = [
    ModelMetadata(
        model_name="DETR_onnx",
        model_group="red",
        op_by_op_backend=OpByOpBackend.STABLEHLO,
        assert_pcc=False,
        assert_atol=False,
        compile_depth=CompileDepth.TTNN_IR,
    )
]


@pytest.mark.parametrize("model_info", DETR_ONNX_VARIANTS, ids=lambda x: x.model_name)
@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
def test_detr_onnx(record_property, model_info, mode, execute_mode):
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    # check if OpByOp
    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
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
        reason="Out of Memory: Not enough space to allocate 59244544 B L1 buffer across 64 banks, where each bank needs to store 925696 B - https://github.com/tenstorrent/tt-torch/issues/729",
        model_group=model_info.model_group,
    )

    tester = ThisTester(
        model_name=model_info.model_name,
        model_info=model_info,
        mode=mode,
        assert_pcc=model_info.assert_pcc,
        assert_atol=model_info.assert_atol,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_info.model_group,
    )
    results = tester.test_model()

    if mode == "eval":
        # Results
        print(results)

    tester.finalize()
