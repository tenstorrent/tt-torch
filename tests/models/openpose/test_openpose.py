# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/lllyasviel/control_v11p_sd15_openpose

import torch
from diffusers.utils import load_image
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend

dependencies = ["controlnet_aux==0.0.9"]


class ThisTester(ModelTester):
    def _load_model(self):
        from controlnet_aux import OpenposeDetector

        model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        image = load_image(
            "https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/input.png"
        )
        arguments = {"input_image": image, "hand_and_face": True}
        return arguments


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.usefixtures("manage_dependencies")
@pytest.mark.skip(reason="failing during torch run with bypass compilation")
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_openpose(record_property, mode, op_by_op):
    model_name = "OpenPose"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name, mode, compiler_config=cc, record_property_handle=record_property
    )
    tester.test_model()
    tester.finalize()
