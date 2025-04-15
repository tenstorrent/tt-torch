# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# https://huggingface.co/black-forest-labs/FLUX.1-schnell
from diffusers import FluxPipeline
from transformers import T5TokenizerFast, T5EncoderModel, CLIPTextModel, CLIPTokenizer
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
import torch


class ThisTester(ModelTester):
    def _load_model(self):
        # Flux Pipeline
        model_info = self.model_name
        pipe = FluxPipeline.from_pretrained(model_info, torch_dtype=torch.bfloat16)

        # CLIP Tokenizer and Text Encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16
        )

        # T5 Tokenizer and Text Encoder
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            "google/t5-v1_1-xxl", torch_dtype=torch.bfloat16
        )
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            "google/t5-v1_1-xxl", torch_dtype=torch.bfloat16
        )

        return pipe

    def _load_inputs(self):
        prompt = [
            "A cat holding a sign that says hello world",
        ]
        max_sequence_length = 256

        """
        Pooled Prompt Embedding - CLIP
        """
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        pooled_prompt_embeds = self.text_encoder(
            text_inputs.input_ids, output_hidden_states=False
        )
        pooled_prompt_embeds = pooled_prompt_embeds.pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=torch.bfloat16)
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(1, -1)

        """
        Prompt Embedding - T5
        """
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder_2(text_input_ids, output_hidden_states=False)[
            0
        ]
        prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(1, seq_len, -1)
        arguments = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "guidance_scale": 0.0,
            "num_inference_steps": 2,
            "max_sequence_length": max_sequence_length,
            "output_type": "latent",
        }
        return arguments


model_info_list = [
    ("flux_schnell", "black-forest-labs/FLUX.1-schnell"),
    ("flux_dev", "black-forest-labs/FLUX.1-dev"),
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
def test_flux(record_property, model_info, mode, op_by_op):
    _, model_name = model_info

    cc = CompilerConfig()
    cc.enable_consteval = False
    cc.consteval_parameters = False
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO
    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=False,
        assert_pcc=False,
        model_group="red",
    )
    results = tester.test_model()
    if mode == "eval":
        image = results.images[0]

    tester.finalize()
