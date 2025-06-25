# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np
from PIL import Image
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.glpn_kitti.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return ModelLoader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        inputs = ModelLoader.load_inputs(dtype_override=torch.bfloat16)
        self.image = ModelLoader.image
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_glpn_kitti(record_property, mode, op_by_op):
    model_name = "GLPN-KITTI"

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
        relative_atol=0.013,
        compiler_config=cc,
        record_property_handle=record_property,
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
