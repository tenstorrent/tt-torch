# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# https://huggingface.co/black-forest-labs/FLUX.1-schnell

from diffusers import FluxPipeline, AutoencoderTiny
from transformers import T5TokenizerFast, T5EncoderModel, CLIPTextModel, CLIPTokenizer
import pytest
import numpy as np
import torch

from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    OpByOpBackend,
    ModelMetadata,
)

# Special subclass for ModelMetadata for Flux models
class FluxModelMetadata(ModelMetadata):
    def __init__(self, model_name, guidance_scale, model_group="red", **kwargs):
        super().__init__(
            model_name=model_name,
            model_group=model_group,
            **kwargs  # Pass through other ModelMetadata params like compile_depth, etc.
        )
        self.guidance_scale = guidance_scale


FLUX_MODELS = [
    FluxModelMetadata(
        model_name="black-forest-labs/FLUX.1-schnell", guidance_scale=0.0
    ),
    FluxModelMetadata(model_name="black-forest-labs/FLUX.1-dev", guidance_scale=3.5),
]


class ThisTester(ModelTester):
    def _load_model(self):
        # Flux Pipeline
        # self.model_name is set by ModelTester's __init__ via model_info.model_name
        # "model_info", FLUX_MODELS, ids=lambda mi: mi.model_name.split("/")[-1]
        pipe = FluxPipeline.from_pretrained(
            self.model_name,  # Uses self.model_name (derived from model_info)
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taef1", torch_dtype=torch.bfloat16
        )
        pipe.enable_attention_slicing()
        pipe.enable_vae_tiling()
        self.transformer = pipe.transformer
        self.scheduler = pipe.scheduler
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.image_processor = pipe.image_processor
        self.pipe = pipe

        return self.transformer

    def _load_inputs(self):
        max_sequence_length = 256
        prompt = "An astronaut riding a horse in a futuristic city"
        # Use guidance_scale from model_info
        do_classifier_free_guidance = self.model_info.guidance_scale > 1.0

        num_inference_steps = 1  # set to 1 for single denoising loop
        max_sequence_length = 256
        height = 128
        width = 128
        num_images_per_prompt = 1

        dtype = torch.bfloat16
        batch_size = 1
        num_channels_latents = self.transformer.config.in_channels // 4
        prompt_2 = prompt
        lora_scale = None
        text_inputs_clip = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_clip = text_inputs_clip.input_ids
        pooled_prompt_embeds = self.text_encoder(
            text_input_ids_clip, output_hidden_states=False
        ).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        text_inputs_t5 = self.tokenizer_2(
            prompt_2,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_t5 = text_inputs_t5.input_ids
        prompt_embeds = self.text_encoder_2(
            text_input_ids_t5, output_hidden_states=False
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        _, seq_len_t5, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len_t5, -1
        )
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)
        height_latent = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))

        shape = (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent,
            width_latent,
        )

        latents = torch.randn(shape, dtype=dtype)
        latents = latents.view(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent // 2,
            2,
            width_latent // 2,
            2,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size * num_images_per_prompt,
            (height_latent // 2) * (width_latent // 2),
            num_channels_latents * 4,
        )

        # Prepare latent image IDs (Flux specific)
        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        (
            latent_image_id_height,
            latent_image_id_width,
            latent_image_id_channels,
        ) = latent_image_ids.shape
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        latent_image_ids = latent_image_ids.to(dtype=dtype)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = (1.15 - 0.5) / (4096 - 256) * (image_seq_len - 256) + 0.5
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, mu=mu)
        timesteps = self.scheduler.timesteps
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        joint_attention_kwargs = {}

        if do_classifier_free_guidance:
            # Use guidance_scale from model_info
            guidance = torch.full([1], self.model_info.guidance_scale, dtype=dtype)
        else:
            guidance = None

        arguments = {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": joint_attention_kwargs,
            "return_dict": False,
        }
        self.latents = latents
        return arguments


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
# Updated parametrization to use model_info from FLUX_MODELS
@pytest.mark.parametrize("model_info", FLUX_MODELS, ids=lambda x: x.model_name)
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
# Updated function signature: removed model_name and guidance_scale, added model_info
def test_flux(record_property, mode, model_info, execute_mode):
    # model_group = "red" # Removed, now comes from model_info.model_group

    cc = CompilerConfig()
    cc.enable_consteval = False
    cc.consteval_parameters = False

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
        model_info.model_name,  # Use model_info.model_name
        bringup_status="FAILED_RUNTIME",
        reason="Out of Memory: Not enough space to allocate 75497472 B DRAM buffer across 12 banks, where each bank needs to store 6291456 B",
        model_group=model_info.model_group,  # Use model_info.model_group
    )

    tester = ThisTester(
        model_info.model_name,  # Pass model_name string from model_info
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=model_info.assert_atol,
        assert_pcc=model_info.assert_pcc,
        model_group=model_info.model_group,
        model_info=model_info,
    )
    results = tester.test_model()
    if mode == "eval":
        noise_pred = results[0]
        latents = tester.scheduler.step(
            noise_pred, tester.scheduler.timesteps[0], tester.latents, return_dict=False
        )[0]
        latents = tester.pipe._unpack_latents(
            latents, 128, 128, tester.pipe.vae_scale_factor
        )
        latents = (
            latents / tester.vae.config.scaling_factor
        ) + tester.vae.config.shift_factor
        image = tester.vae.decode(latents, return_dict=False)[0].detach()
        image = tester.image_processor.postprocess(image, output_type="pil")[0]
        # image.save("astronaut.png")

    tester.finalize()
