# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Phi-3.5-vision: https://huggingface.co/microsoft/Phi-3.5-vision-instruct

import pytest
import requests

from transformers import AutoProcessor, AutoModelForCausalLM
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from PIL import Image
from io import BytesIO


class ThisTester(ModelTester):
    def _load_model(self):
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
def test_phi_3p5_vision(record_property, model_name, mode, op_by_op):
    model_group = "red"
    cc = CompilerConfig()
    cc.enable_consteval = True

    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_FE_COMPILATION",
        model_group=model_group,
        reason="Can't lower remainder op in TORCH IR --> TORCH BACKEND IR",
    )

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        is_token_output=True,
        model_group=model_group,
        run_generate=True,
    )

    results = tester.test_model(assert_eval_token_mismatch=False)

    if mode == "eval":
        decoded_output = tester.processor.batch_decode(
            results[:, tester.inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        print(
            f"""
        model_name: {model_name}
        input: {tester.messages}
        output: {decoded_output}
        """
        )
    tester.finalize()
