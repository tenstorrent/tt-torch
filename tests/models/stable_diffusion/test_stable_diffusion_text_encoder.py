# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# https://huggingface.co/docs/diffusers/v0.33.1/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline
import torch
import pytest
from tests.utils import ModelTester
from diffusers import StableDiffusion3Pipeline
from transformers import CLIPTextModel, CLIPTokenizer
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        model_path = self.model_name
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer"
        )
        return self.text_encoder

    def _load_inputs(self):
        prompt = "a photo of an astronaut riding a horse on mars"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        # Optionally handle attention_mask as in the pipeline
        attention_mask = None
        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = inputs["attention_mask"]
        return {"input_ids": inputs["input_ids"], "attention_mask": attention_mask}


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_stable_diffusion_text_encoder(record_property, mode, op_by_op):
    model_name = "CompVis/stable-diffusion-v1-4"
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
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
    )
    results = tester.test_model()
    tester.finalize()
