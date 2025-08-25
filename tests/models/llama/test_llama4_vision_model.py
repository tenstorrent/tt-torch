# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn  # Add this line
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from transformers import AutoConfig
from transformers.models.llama4.modeling_llama4 import Llama4VisionModel


# Temporary monkeypatch to avoid complex tensors, need to revisit this
class MockVisionRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size = config.image_size
        patch_size = config.patch_size
        self.num_patches = (image_size // patch_size) ** 2 + 1
        self.head_dim = config.hidden_size // config.num_attention_heads

    def forward(self, pixel_values):
        # CRITICAL: Create tensor with float32 dtype to avoid XLA issues
        return torch.ones(
            self.num_patches,
            self.num_patches,
            self.head_dim,
            device=pixel_values.device,
            dtype=torch.float32,
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
            intermediate_size=16,  # After pixel shuffle: 64/(2²) = 16
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=64,
            patch_size=16,
            num_channels=3,
            attention_dropout=0.0,
            rope_theta=getattr(original_vision_config, "rope_theta", 10000.0),
            vision_output_dim=64,
            projector_input_dim=32,  # fc1: 16 → 32
            projector_output_dim=32,  # fc2: 32 → 32
            pixel_shuffle_ratio=2,
            projector_dropout=0.0,
        )

        # Create vision-only model
        self.model = Llama4VisionModel(vision_config)
        # self.model = self.model.to(torch.bfloat16)
        self.model.eval()

        # Replace vision rotary embedding
        self.model.rotary_embedding = MockVisionRotaryEmbedding(vision_config)

        # CRITICAL: Also mock the vision_apply_rotary_emb function
        import transformers.models.llama4.modeling_llama4 as llama4_mod

        def mock_vision_apply_rotary_emb(query, key, freqs_ci):
            # Just return inputs unchanged (no complex tensor operations)
            return query, key

        # Monkey patch the module-level function
        llama4_mod.vision_apply_rotary_emb = mock_vision_apply_rotary_emb

        return self.model

    def _load_inputs(self):
        # Create dummy image inputs
        # Standard vision input: pixel values with shape [batch, channels, height, width]
        vision_config = self.model.config

        batch_size = 1
        channels = vision_config.num_channels  # Usually 3 (RGB)
        height = vision_config.image_size  # e.g., 448
        width = vision_config.image_size

        pixel_values = torch.randn(
            batch_size, channels, height, width, dtype=torch.bfloat16
        )

        # CRITICAL: Ensure ALL tensors are float type and on same device
        # This prevents "CPULongType" XLA tensor errors
        pixel_values = pixel_values.float()  # Convert to float32 for XLA compatibility

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
        "llama4_vision_only",  # Custom name since we're not using standard model loading
        mode,
        compiler_config=cc,
        assert_atol=False,
        assert_pcc=True,
        required_pcc=0.71,
        record_property_handle=record_property,
        # backend="tt",
    )

    results = tester.test_model()
    tester.finalize()
