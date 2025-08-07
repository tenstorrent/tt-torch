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


class TextEncoderTester(ModelTester):
    def _load_model(self):
        model_dict = dict(model_info_list)
        model_path = model_dict[self.model_name]
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        # Return only the text encoder for compilation
        return self.pipe.text_encoder_3

    def _load_inputs(self):
        prompt = "A futuristic cityscape at sunset"

        # Tokenize the prompt
        text_inputs = self.pipe.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer_3.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        arguments = {
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
        }

        return arguments


model_info_list = [
    ("SD3.5-medium-text-encoder", "stabilityai/stable-diffusion-3.5-medium"),
    ("SD3.5-large-text-encoder", "stabilityai/stable-diffusion-3.5-large"),
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
def test_stable_diffusion_text_encoder(record_property, model_info, mode, op_by_op):
    model_group = "red"
    model_name, model_path = model_info

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = TextEncoderTester(
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
