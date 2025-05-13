# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from diffusers import StableDiffusion3Pipeline, AutoencoderTiny

from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        model_info = self.model_name
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_info,
            text_encoder_3=None,
            tokenizer_3=None,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        # memory optimization recommended by: https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_3#tiny-autoencoder-for-stable-diffusion-3
        pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd3", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        pipe.enable_attention_slicing()
        return pipe

    def _load_inputs(self):
        prompt = [
            "a photo of an astronaut riding a horse on mars",
        ]

        negative_prompt = ""
        height = 512
        width = 512
        guidance_scale = 7.0
        arguments = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
        }

        return arguments


model_info_list = [
    ("SD3.5-medium", "stabilityai/stable-diffusion-3.5-medium"),
    ("SD3.5-large", "stabilityai/stable-diffusion-3.5-large"),
]


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_info",
    model_info_list,
    ids=[model_info[0] for model_info in model_info_list],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_stable_diffusion_3_5(record_property, model_info, mode, op_by_op):
    _, model_name = model_info

    cc = CompilerConfig()
    cc.enable_consteval = True
    # consteval_parameters is disabled because it results in a memory related crash
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    # skip_full_eval_test(
    #     record_property,
    #     cc,
    #     model_name,
    #     bringup_status="FAILED_RUNTIME",
    #     reason="Model encounters 'Fatal Python error: Aborted' due to running out of memory during compilation.",
    #     model_group=model_group,
    #     model_name_filter="stabilityai/stable-diffusion-3.5-medium",
    # )

    # skip_full_eval_test(
    #     record_property,
    #     cc,
    #     model_name,
    #     bringup_status="FAILED_RUNTIME",
    #     reason="Model encounters 'Fatal Python error: Aborted' due to running out of memory during compilation.",
    #     model_group=model_group,
    #     model_name_filter="stabilityai/stable-diffusion-3.5-large",
    # )

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=False,
        assert_pcc=False,
        model_group="red",
    )
    results = tester.test_model()
    if mode == "eval":
        image = results.images[0]
        image.save("generated_image.png")

    tester.finalize()
