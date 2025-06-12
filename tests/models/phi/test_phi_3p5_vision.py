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


class ThisTester(ModelTester):
    def _load_model(self):
        # Use AutoProcessor for multimodal models
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True  # Important for Phi-3 models
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,  # Ensure bfloat16
            trust_remote_code=True,  # Important for Phi-3 models
            attn_implementation="eager",  # Use eager attention for compatibility
        )
        return model

    def _load_inputs(self):
        """# Example image URL and prompt structure
        image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
        # Store prompt for later use/display
        self.current_prompt_text = "<|user|>\n<|image_1|>\nWhat animal is on the candy?<|end|>\n<|assistant|>\n"

        raw_image = Image.open(requests.get(image_url, stream=True).raw)

        # Process both text and image using the AutoProcessor
        inputs = self.processor(
            text=self.current_prompt_text, images=raw_image, return_tensors="pt"
        )

        # Ensure pixel_values are bfloat16
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # The ModelTester expects a dictionary of tensors, which processor already provides
        return inputs"""
        pipe = pipeline(
            "image-text-to-text",
            model="microsoft/Phi-3.5-vision-instruct",
            trust_remote_code=True,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                    },
                    {"type": "text", "text": "What animal is on the candy?"},
                ],
            },
        ]
        return pipe(text=messages)


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
