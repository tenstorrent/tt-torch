# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn  # Add this line
import pytest
from typing import Tuple
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from transformers import AutoConfig
from transformers.models.llama4.modeling_llama4 import Llama4VisionModel


def reshape_for_broadcast(freqs: torch.Tensor, query: torch.Tensor):
    """Helper function to reshape frequency tensors for broadcasting"""
    ndim = query.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(query.shape)]
    return freqs.view(*shape)


def real_valued_vision_apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_ci: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embedding using real-valued arithmetic instead of complex numbers.

    This implements the same rotation as complex multiplication but using only real tensors:
    x' = x * cos(θ) - y * sin(θ)
    y' = x * sin(θ) + y * cos(θ)

    Args:
        query: Query tensor
        key: Key tensor
        freqs_ci: Tuple of (freqs_cos, freqs_sin) tensors
    """
    # Extract cos and sin from the tuple
    freqs_cos, freqs_sin = freqs_ci

    # Reshape query and key to separate real/imaginary parts
    query_real, query_imag = query.float().chunk(2, dim=-1)
    key_real, key_imag = key.float().chunk(2, dim=-1)

    # Reshape frequency tensors for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, query_real).to(query_real.device)
    freqs_sin = reshape_for_broadcast(freqs_sin, query_real).to(query_real.device)

    # Apply rotation using real arithmetic
    query_out_real = query_real * freqs_cos - query_imag * freqs_sin
    query_out_imag = query_real * freqs_sin + query_imag * freqs_cos
    key_out_real = key_real * freqs_cos - key_imag * freqs_sin
    key_out_imag = key_real * freqs_sin + key_imag * freqs_cos

    # Concatenate back to original shape
    query_out = torch.cat([query_out_real, query_out_imag], dim=-1)
    key_out = torch.cat([key_out_real, key_out_imag], dim=-1)

    return query_out.type_as(query), key_out.type_as(key)


# Real-valued rotary embedding to avoid complex tensors
class RealValuedVisionRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        idx = config.image_size // config.patch_size
        img_idx = torch.arange(idx**2, dtype=torch.int32).reshape(idx**2, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = -2  # ID_CLS_TOKEN
        frequencies_x = img_idx % idx  # get the coordinates of the 2d matrix along x
        frequencies_y = img_idx // idx  # get the coordinates of the 2d matrix along y
        freq_dim = config.hidden_size // config.num_attention_heads // 2
        rope_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, freq_dim, 2)[: (freq_dim // 2)].float() / freq_dim)
        )
        freqs_x = (
            (frequencies_x + 1)[..., None] * rope_freq[None, None, :]
        ).repeat_interleave(2, dim=-1)
        freqs_y = (
            (frequencies_y + 1)[..., None] * rope_freq[None, None, :]
        ).repeat_interleave(2, dim=-1)
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)

        # Store cos and sin separately instead of as complex numbers
        self.freqs_cos = torch.cos(freqs)
        self.freqs_sin = torch.sin(freqs)

    def forward(self, pixel_values):
        return self.freqs_cos.to(pixel_values.device), self.freqs_sin.to(
            pixel_values.device
        )


class ThisTester(ModelTester):
    def _load_model(self):
        model_name = "meta-llama/Llama-4-Scout-17B-16E"

        # Get the full config but only use vision part
        full_config = AutoConfig.from_pretrained(model_name)
        original_vision_config = full_config.vision_config

        # Create a NEW vision config with smaller dimensions
        from transformers.models.llama4.configuration_llama4 import Llama4VisionConfig

        vision_config = Llama4VisionConfig(
            hidden_size=64,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=64,
            patch_size=16,
            num_channels=3,
            attention_dropout=0.0,
            rope_theta=getattr(original_vision_config, "rope_theta", 10000.0),
            vision_output_dim=64,
            projector_input_dim=32,
            projector_output_dim=32,
            pixel_shuffle_ratio=2,
            projector_dropout=0.0,
        )

        self.model = Llama4VisionModel(vision_config)
        self.model.eval()

        ## UNCOMMENT BELOW TO ENABLE REAL-VALUED ROTARY EMB

        # # Replace vision rotary embedding with real-valued version
        # self.model.rotary_embedding = RealValuedVisionRotaryEmbedding(vision_config)

        # import transformers.models.llama4.modeling_llama4 as llama4_mod

        # # Monkey patch the module-level function to use real-valued version
        # llama4_mod.vision_apply_rotary_emb = real_valued_vision_apply_rotary_emb

        return self.model

    def _load_inputs(self):
        # Create dummy image inputs
        # Standard vision input: pixel values with shape [batch, channels, height, width]
        vision_config = self.model.config

        batch_size = 1
        channels = vision_config.num_channels
        height = vision_config.image_size
        width = vision_config.image_size

        pixel_values = torch.randn(
            batch_size, channels, height, width, dtype=torch.float32
        )

        inputs = {
            "pixel_values": pixel_values,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_llama4_vision_layer(record_property, mode, op_by_op):
    cc = CompilerConfig()
    cc.enable_consteval = False
    cc.consteval_parameters = False
    cc.dump_info = True

    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        "llama4_vision_only",
        mode,
        compiler_config=cc,
        assert_atol=False,
        assert_pcc=True,
        required_pcc=0.71,
        record_property_handle=record_property,
        backend="tt",
    )

    results = tester.test_model()
    tester.finalize()
