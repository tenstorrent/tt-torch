# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Phi-3.5-vision: https://huggingface.co/microsoft/Phi-3.5-vision-instruct

import torch
import pytest
import requests

from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file


class ThisTester(ModelTester):
    def _load_model(self):
        # Use AutoProcessor for multimodal models
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        self.tokenizer = self.processor.tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,  # Ensure bfloat16
            trust_remote_code=True,
            _attn_implementation="eager",
        )
        return model

    def _load_inputs(self):
        image_file = get_file(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
        )
        image = Image.open(str(image_file))

        messages = [
            {"role": "user", "content": "<|image_1|>\nWhat is the cat wearing?"}
        ]

        # 3. Apply the chat template and process inputs
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            prompt,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "microsoft/Phi-3.5-vision-instruct",
    ],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_phi(record_property, model_name, mode, op_by_op):
    model_group = "red"
    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        is_token_output=True,
        model_group=model_group,
    )

    results = tester.test_model(assert_eval_token_mismatch=False)

    if mode == "eval":
        decoded_output = tester.tokenizer.decode(results[0])
        print(
            f"""
        model_name: {model_name}
        input: {tester.test_input}
        output: {decoded_output}
        """
        )
    tester.finalize()
