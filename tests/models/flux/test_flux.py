# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# https://huggingface.co/black-forest-labs/FLUX.1-schnell

import pytest
import torch
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.flux.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs(dtype_override=torch.bfloat16)


# Print available variants for reference
available_variants = ModelLoader.query_available_variants()
print("Available variants: ", [str(k) for k in available_variants.keys()])


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "variant,variant_config",
    available_variants.items(),
    ids=[str(k) for k in available_variants.keys()],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_flux(record_property, variant, variant_config, mode, op_by_op):
    cc = CompilerConfig()
    cc.enable_consteval = True
    # consteval_parameters is disabled because it results in a memory related crash
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    loader = ModelLoader(variant=variant)
    model_info = loader.get_model_info(variant=variant)

    skip_full_eval_test(
        record_property,
        cc,
        model_info.name,
        bringup_status="FAILED_RUNTIME",
        reason="Out of Memory: Not enough space to allocate 75497472 B DRAM buffer across 12 banks, where each bank needs to store 6291456 B",
        model_group=model_info.group,
    )

    tester = ThisTester(
        model_info.name,
        mode,
        loader=loader,
        model_info=model_info,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=False,
        assert_pcc=False,
    )

    results = tester.test_model()
    if mode == "eval":
        noise_pred = results[0]

        # Access pipeline components through loader
        pipe = loader.pipe
        latents_input = tester._load_inputs()["hidden_states"]

        # Set up scheduler (simplified version)
        import numpy as np

        sigmas = np.linspace(1.0, 1.0, 1)  # Single step
        image_seq_len = latents_input.shape[1]
        mu = (1.15 - 0.5) / (4096 - 256) * (image_seq_len - 256) + 0.5
        pipe.scheduler.set_timesteps(1, sigmas=sigmas, mu=mu)

        latents = pipe.scheduler.step(
            noise_pred, pipe.scheduler.timesteps[0], latents_input, return_dict=False
        )[0]

        latents = pipe._unpack_latents(latents, 128, 128, pipe.vae_scale_factor)
        latents = (
            latents / pipe.vae.config.scaling_factor
        ) + pipe.vae.config.shift_factor

        image = pipe.vae.decode(latents, return_dict=False)[0].detach()
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        # image.save("astronaut.png")

    tester.finalize()
