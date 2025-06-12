# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Phi-3.5-vision: https://huggingface.co/microsoft/Phi-3.5-vision-instruct

<<<<<<< HEAD
import pytest
import requests

from transformers import AutoProcessor, AutoModelForCausalLM
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from PIL import Image
from io import BytesIO
=======
import torch
import pytest
import requests

from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from PIL import Image
>>>>>>> 6bf3f62e (need to figure out the vision model test. Not sure if im doing it well. FIXING SOME REBASE ERRORS.)


class ThisTester(ModelTester):
    def _load_model(self):
<<<<<<< HEAD
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, _attn_implementation="eager"
        )

        return self.model

    def _load_inputs(self):
        image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        image = Image.open(BytesIO(requests.get(image_url).content))
        self.messages = [
            {"role": "user", "content": "<|image_1|>\nWhat is this image about?"},
        ]

        prompt = self.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        self.inputs = self.processor(prompt, [image], return_tensors="pt").to(
            self.model.device
        )
        arguments = {
            **self.inputs,
            "use_cache": False,
            "max_new_tokens": 20,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        return arguments
=======
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
>>>>>>> 6bf3f62e (need to figure out the vision model test. Not sure if im doing it well. FIXING SOME REBASE ERRORS.)


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
<<<<<<< HEAD
def test_phi_3p5_vision(record_property, model_name, mode, op_by_op):
    model_group = "red"
    cc = CompilerConfig()
    cc.enable_consteval = True

=======
def test_phi(record_property, model_name, mode, op_by_op):
    model_group = "red"
    cc = CompilerConfig()
>>>>>>> 6bf3f62e (need to figure out the vision model test. Not sure if im doing it well. FIXING SOME REBASE ERRORS.)
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

<<<<<<< HEAD
    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_FE_COMPILATION",
        model_group=model_group,
        reason="Can't lower remainder op in TORCH IR --> TORCH BACKEND IR",
    )

=======
>>>>>>> 6bf3f62e (need to figure out the vision model test. Not sure if im doing it well. FIXING SOME REBASE ERRORS.)
    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        is_token_output=True,
        model_group=model_group,
<<<<<<< HEAD
        run_generate=True,
=======
>>>>>>> 6bf3f62e (need to figure out the vision model test. Not sure if im doing it well. FIXING SOME REBASE ERRORS.)
    )

    results = tester.test_model(assert_eval_token_mismatch=False)

    if mode == "eval":
<<<<<<< HEAD
        decoded_output = tester.processor.batch_decode(
            results[:, tester.inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        print(
            f"""
        model_name: {model_name}
        input: {tester.messages}
=======
        decoded_output = tester.tokenizer.decode(results[0])
        print(
            f"""
        model_name: {model_name}
        input: {tester.test_input}
>>>>>>> 6bf3f62e (need to figure out the vision model test. Not sure if im doing it well. FIXING SOME REBASE ERRORS.)
        output: {decoded_output}
        """
        )
    tester.finalize()
