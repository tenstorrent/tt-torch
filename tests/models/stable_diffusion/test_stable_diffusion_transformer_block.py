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

        transformer = SD3Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        return transformer.transformer_blocks[0]

    def _load_inputs(self):
        hidden_states = torch.randn(1, 1024, 1536)
        encoder_hidden_states = torch.randn(1, 333, 1536)
        temb = torch.rand(1, 1536)
        hidden_states = hidden_states.to(torch.bfloat16)
        encoder_hidden_states = encoder_hidden_states.to(torch.bfloat16)
        temb = temb.to(torch.bfloat16)
        joint_attention_kwargs = {}

        arguments = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "temb": temb,
            "joint_attention_kwargs": joint_attention_kwargs,
        }
        return arguments


model_info_list = [
    ("SD3.5-medium-transformer", "stabilityai/stable-diffusion-3.5-medium"),
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
def test_stable_diffusion_transformer(record_property, model_info, mode):
    model_group = "red"
    model_name, model_path = model_info

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

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
