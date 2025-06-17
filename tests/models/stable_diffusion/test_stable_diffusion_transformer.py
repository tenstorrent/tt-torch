# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester, skip_full_eval_test
from diffusers import (
    StableDiffusion3Pipeline,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
)

from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        model_dict = dict(model_info_list)
        model_path = model_dict[self.model_name]
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        self.transformer = SD3Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        return self.transformer

    def _load_inputs(self):
        prompt = "A futuristic cityscape at sunset"
        num_images_per_prompt = 1
        height = 512
        width = 512
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,  # Using the same prompt for all encoders for simplicity
            prompt_3=prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=False,  # For simplicity in this direct call
        )

        num_channels_latents = self.transformer.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size=num_images_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=None,
            generator=None,
        )

        self.scheduler.set_timesteps(28)
        timestep = self.scheduler.timesteps[0].expand(latents.shape[0])
        arguments = {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "joint_attention_kwargs": {},
            "return_dict": False,
        }
        return arguments


model_info_list = [
    ("SD3.5-medium-transformer", "stabilityai/stable-diffusion-3.5-medium"),
    ("SD3.5-large-transformer", "stabilityai/stable-diffusion-3.5-large"),
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
def test_stable_diffusion_transformer(record_property, model_info, mode, op_by_op):
    model_group = "red"
    model_name, model_path = model_info

    cc = CompilerConfig()
    cc.enable_consteval = True
    # consteval_parameters is disabled because it results in a memory related crash
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_RUNTIME",
        reason="Model compilation exceeds CI timeout limit (240 minutes), compiles e2e locally. Model execution fails with `Not enough space to allocate 47316992 B DRAM buffer across 12 banks.`",
        model_group=model_group,
        model_name_filter="SD3.5-large-transformer",
    )

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=False,
        assert_pcc=True,
        model_group=model_group,
    )
    results = tester.test_model()

    tester.finalize()
