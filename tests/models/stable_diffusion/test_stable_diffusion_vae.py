# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester, skip_full_eval_test
from diffusers import StableDiffusion3Pipeline

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

        # Store VAE config for input dimensions
        self.vae = self.pipe.vae

        # Test VAE decoder by default (most common use case)
        if "encoder" in self.model_name:
            return self.vae.encoder
        else:  # decoder
            return self.vae.decoder

    def _load_inputs(self):
        if "encoder" in self.model_name:
            # VAE Encoder: takes RGB images, outputs latents
            batch_size = 1
            channels = self.vae.config.in_channels  # Should be 3 for RGB
            height = 512
            width = 512

            # Create sample RGB image (normalize to [-1, 1] range like real images)
            sample = (
                torch.randn(batch_size, channels, height, width, dtype=torch.bfloat16)
                * 2.0
                - 1.0
            )  # Scale to [-1, 1]

            arguments = {
                "sample": sample,
            }
        else:
            # VAE Decoder: takes latents, outputs RGB images
            batch_size = 1
            latent_channels = self.vae.config.latent_channels  # Likely 4 or 16
            # Calculate latent dimensions based on scaling factor
            latent_height = 512 // 8  # Typical VAE downsampling factor
            latent_width = 512 // 8

            # Create sample latents
            sample = torch.randn(
                batch_size,
                latent_channels,
                latent_height,
                latent_width,
                dtype=torch.bfloat16,
            )

            # Apply scaling factor if present
            if hasattr(self.vae.config, "scaling_factor"):
                sample = sample * self.vae.config.scaling_factor

            arguments = {
                "sample": sample,
            }

        return arguments


model_info_list = [
    ("SD3.5-medium-vae-decoder", "stabilityai/stable-diffusion-3.5-medium"),
    ("SD3.5-medium-vae-encoder", "stabilityai/stable-diffusion-3.5-medium"),
    ("SD3.5-large-vae-decoder", "stabilityai/stable-diffusion-3.5-large"),
    ("SD3.5-large-vae-encoder", "stabilityai/stable-diffusion-3.5-large"),
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
def test_stable_diffusion_vae(record_property, model_info, mode, op_by_op):
    model_group = "red"
    model_name, model_path = model_info

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True  # will run into OOM error faster if this is disabled
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    if model_name == "SD3.5-medium-vae-encoder":
        reason = "Out of Memory: Not enough space to allocate 84213760 B L1 buffer across 64 banks, where each bank needs to store 1315840 B"
    elif model_name == "SD3.5-medium-vae-decoder":
        reason = "Out of Memory: Not enough space to allocate 71860224 B L1 buffer across 64 banks, where each bank needs to store 1122816 B"
    else:
        reason = (
            "medium-vae encounters OOM errors, so skipping large-vae full eval test"
        )

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_RUNTIME",
        reason=reason,
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
