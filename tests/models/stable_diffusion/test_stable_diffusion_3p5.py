# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# https://huggingface.co/runwayml/stable-diffusion-v1-5
from diffusers import DiffusionPipeline
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        model_id = "stabilityai/stable-diffusion-3.5-medium"
        pipe = DiffusionPipeline.from_pretrained(model_id)
        return pipe

    def _load_inputs(self):
        prompt = [
            "a photo of an astronaut riding a horse on mars",
        ]
        return prompt


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
# @pytest.mark.parametrize(
#     "op_by_op",
#     [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
#     ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
# )
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.TORCH],
    ids=["op_by_op_torch"],
)
# @pytest.mark.xfail(
#     reason="Fails due to pt2 compile issue when finishing generation, but we can still generate a graph"
# )
def test_stable_diffusion_3p5(record_property, mode, op_by_op):
    model_name = "Stable Diffusion 3.5 Medium"

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
    results = tester.test_model()
    if mode == "eval":
        image = results.images[0]

    tester.finalize()
