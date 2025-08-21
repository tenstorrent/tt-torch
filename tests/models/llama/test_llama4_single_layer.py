# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from transformers import AutoTokenizer, Llama4ForConditionalGeneration, AutoConfig


class ThisTester(ModelTester):
    def _load_model(self):
        model_name = "meta-llama/Llama-4-Scout-17B-16E"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        config = AutoConfig.from_pretrained(model_name)
        config.vision_config.num_hidden_layers = 1
        config.text_config.num_hidden_layers = 4

        # Disable RoPE for all layers
        # config.text_config.no_rope_layers = [False] * config.text_config.num_hidden_layers

        self.model = Llama4ForConditionalGeneration.from_pretrained(
            model_name,
            config=config,
            # device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Temporary monkeypatch to avoid complex tensors, need to revisit this
        # def mock_rotary_forward(x, position_ids):
        #     # Return identity - no rotation applied
        #     batch_size, seq_len = position_ids.shape
        #     head_dim = x.shape[-1] // self.model.language_model.model.config.num_attention_heads
        #
        #     # Create a dummy tensor that looks like complex but is just real
        #     # This will be ignored since we disabled RoPE usage in layers
        #     dummy_rotation = torch.ones(
        #         batch_size, seq_len, head_dim,
        #         device=x.device,
        #         dtype=x.dtype
        #     )
        #     return dummy_rotation
        #
        # # Replace the rotary embedding forward method
        # self.model.language_model.model.rotary_emb.forward = mock_rotary_forward

        return self.model

    def _load_inputs(self):
        messages = [
            {"role": "user", "content": "Who are you?"},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(torch.bfloat16)
        # Disable caching to avoid errors when lowering model
        inputs["use_cache"] = False
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
def test_llama4(record_property, mode, op_by_op):
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.dump_info = True

    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        "meta-llama/Llama-4-Scout-17B-16E",
        mode,
        compiler_config=cc,
        assert_atol=False,
        assert_pcc=True,
        record_property_handle=record_property,
        run_generate=False,  # Disable generation for this test
        backend="tt",
    )
    results = tester.test_model()
    tester.finalize()
