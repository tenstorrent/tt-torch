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


class TextOnlyTester(ModelTester):
    def _load_model(self):
        model_name = "meta-llama/Llama-4-Scout-17B-16E"

        # Get the full config but only use text part
        full_config = AutoConfig.from_pretrained(model_name)
        text_config = full_config.text_config

        # Make model as small as possible for testing
        text_config.num_hidden_layers = 1  # Just test one text layer
        text_config.hidden_size = 512  # Reduce size
        text_config.num_attention_heads = 8
        text_config.num_key_value_heads = 8
        text_config.intermediate_size = 1024
        text_config.vocab_size = 1000  # Reduce vocab size

        # CRITICAL: Fix padding_idx to be within vocab_size range
        text_config.pad_token_id = 0  # Set to valid range or None

        # Disable RoPE for all layers to avoid complex tensors
        text_config.no_rope_layers = [False] * text_config.num_hidden_layers

        # Create text-only model
        self.model = Llama4TextModel(text_config)
        self.model.eval()

        return self.model

    def _load_inputs(self):
        # Create simple text inputs
        text_config = self.model.config

        batch_size = 1
        seq_len = 8  # Short sequence
        vocab_size = text_config.vocab_size

        # Create dummy token IDs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create attention mask
        attention_mask = torch.ones(batch_size, seq_len)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,  # Disable caching to avoid errors when lowering model
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
def test_llama4_text_layer(record_property, mode, op_by_op):

    cc = CompilerConfig()
    cc.enable_consteval = False
    cc.consteval_parameters = False
    cc.dump_info = True

    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = TextOnlyTester(
        "llama4_text_only",
        mode,
        compiler_config=cc,
        assert_atol=False,
        assert_pcc=True,
        record_property_handle=record_property,
        backend="tt",
    )

    results = tester.test_model()
    tester.finalize()
