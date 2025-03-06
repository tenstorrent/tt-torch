# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# https://huggingface.co/runwayml/stable-diffusion-v1-5
from diffusers import StableDiffusionPipeline
import torch
import pytest
from tests.utils import ModelTester


class ThisTester(ModelTester):
    def _load_model(self):
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        return pipe

    def _load_inputs(self):
        prompt = "a photo of an astronaut riding a horse on mars"
        return prompt


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.skip(reason="Dynamo cannot support pipeline.")
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_stable_diffusion(record_property, mode, op_by_op):
    model_name = "Stable Diffusion"

    tester = ThisTester(model_name, mode, record_property_handle=record_property)
    results = tester.test_model()
    if mode == "eval":
        image = results.images[0]

    tester.finalize()
