# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    LMSDiscreteScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        # Load the pre-trained model and tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="unet",
            torch_dtype=torch.bfloat16,
        )
        self.scheduler = LMSDiscreteScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )
        return unet

    def _load_inputs(self):
        # Prepare the text prompt
        prompt = "A fantasy landscape with mountains and rivers"
        text_input = self.tokenizer(prompt, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids)[0]

        # Generate noise
        batch_size = text_embeddings.shape[0]
        height, width = 512, 512  # Output image size
        latents = torch.randn(
            (batch_size, self.framework_model.in_channels, height // 8, width // 8)
        )

        # Set number of diffusion steps
        num_inference_steps = 1
        self.scheduler.set_timesteps(num_inference_steps)

        # Scale the latent noise to match the model's expected input
        latents = latents * self.scheduler.init_noise_sigma

        # Get the model's predicted noise
        latent_model_input = self.scheduler.scale_model_input(latents, 0)
        arguments = {
            "sample": latent_model_input.to(torch.bfloat16),
            "timestep": 0,
            "encoder_hidden_states": text_embeddings.to(torch.bfloat16),
        }
        return arguments


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_stable_diffusion_unet(record_property, mode, op_by_op):
    model_name = "Stable Diffusion UNET"

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
        assert_pcc=True,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()
    if mode == "eval":
        noise_pred = results.sample

    tester.finalize()
