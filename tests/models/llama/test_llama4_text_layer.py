# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from transformers import AutoConfig
from transformers.models.llama4.modeling_llama4 import Llama4TextModel


# Temporary monkeypatch to avoid complex tensors, need to revisit this
class MockTextRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads

    def forward(self, x, position_ids):
        batch_size, seq_len = position_ids.shape
        # Return real tensor instead of complex
        dummy_freqs = torch.ones(
            batch_size, seq_len, self.head_dim, device=x.device, dtype=torch.float32
        )
        return dummy_freqs


class ThisTester(ModelTester):
    def _load_model(self):
        model_name = "meta-llama/Llama-4-Scout-17B-16E"

        # Get the full config but only use text part
        full_config = AutoConfig.from_pretrained(model_name)
        original_text_config = full_config.text_config

        # Create a NEW text config with smaller dimensions
        from transformers.models.llama4.configuration_llama4 import Llama4TextConfig

        text_config = Llama4TextConfig(
            hidden_size=64,
            intermediate_size=256,  # 4x hidden_size
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=1000,
            max_position_embeddings=128,
            pad_token_id=0,
            attention_dropout=0.0,
            rope_theta=getattr(original_text_config, "rope_theta", 10000.0),
        )

        # Create text-only model
        self.model = Llama4TextModel(text_config)
        self.model.eval()

        # CRITICAL: Mock torch.arange to return float32 instead of long
        original_arange = torch.arange

        def mock_arange(*args, **kwargs):
            # Force dtype to float32 for XLA compatibility
            if "dtype" not in kwargs:
                kwargs["dtype"] = torch.float32
            elif kwargs["dtype"] == torch.long:
                kwargs["dtype"] = torch.float32
            return original_arange(*args, **kwargs)

        torch.arange = mock_arange

        # Replace text rotary embedding
        self.model.rotary_emb = MockTextRotaryEmbedding(text_config)

        # CRITICAL: Mock the apply_rotary_emb function for text
        import transformers.models.llama4.modeling_llama4 as llama4_mod

        def mock_apply_rotary_emb(xq, xk, freqs_cis):
            # Just return inputs unchanged (no complex tensor operations)
            return xq, xk

        # Monkey patch the module-level function
        llama4_mod.apply_rotary_emb = mock_apply_rotary_emb

        return self.model.layers[0]  # Return single layer like vision test

    def _load_inputs(self):
        # Get config from the single layer's parent model
        text_config = self.model.config

        batch_size = 1
        seq_len = 8
        hidden_size = text_config.hidden_size

        # Create pre-embedded hidden states (what a single layer expects)
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size, dtype=torch.float32
        )

        # Create attention mask - ensure it's float32, not bool
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)

        # Provide explicit position_ids as float32 to prevent internal arange calls
        position_ids = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)

        # Return as tuple with positional args first, then kwargs
        inputs = (
            hidden_states,  # First positional argument
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,  # Provide explicit float32 position_ids
                "output_attentions": False,
                "use_cache": False,
                "return_dict": True,
            },
        )
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
def test_llama4_text_layer(record_property, mode, op_by_op):
    cc = CompilerConfig()
    cc.enable_consteval = False
    cc.consteval_parameters = False
    cc.dump_info = True

    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        "llama4_text_layer_only",  # Custom name since we're not using standard model loading
        mode,
        compiler_config=cc,
        assert_atol=False,
        assert_pcc=True,
        record_property_handle=record_property,
        # backend="tt",
    )

    results = tester.test_model()
    tester.finalize()
