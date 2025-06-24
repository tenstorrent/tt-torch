# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np
from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, ModelMetadata
from third_party.tt_forge_models.tools.utils import get_file


class ThisTester(ModelTester):
    def _load_model(self):
        self.processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-kitti")
        model = GLPNForDepthEstimation.from_pretrained(
            "vinvino02/glpn-kitti", torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.image = Image.open(str(image_file))
        # prepare image for the model
        inputs = self.processor(images=self.image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs


GLPN_KITTI_VARIANTS = [
    ModelMetadata(
        model_name="GLPN-KITTI",
        relative_atol=0.013,
        model_group="red",
        compile_depth=CompileDepth.TTNN_IR,
    )
]


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize("model_info", GLPN_KITTI_VARIANTS, ids=lambda x: x.model_name)
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
def test_glpn_kitti(record_property, model_info, mode, execute_mode):
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.op_by_op_backend = model_info.op_by_op_backend
    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
        cc.compile_depth = execute_mode
    else:
        cc.compile_depth = model_info.compile_depth

    tester = ThisTester(
        model_name=model_info.model_name,
        model_info=model_info,
        mode=mode,
        relative_atol=model_info.relative_atol,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_info.model_group,
    )
    results = tester.test_model()
    if mode == "eval":
        predicted_depth = results.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=tester.image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # visualize the prediction
        output = prediction.squeeze().cpu().to(float).numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)

    tester.finalize()
