# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/dandelin/vilt-b32-finetuned-vqa

from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import pytest
from tests.utils import ModelTester
import torch
<<<<<<< HEAD
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file
=======
from tt_torch.tools.utils import CompilerConfig, CompileDepth, ModelMetadata
>>>>>>> 1031803 (refactored test_qwen2_token_classification.py, test_timm_image_classification.py, test_vilt.py for the new pytest infra.)


class ThisTester(ModelTester):
    def _load_model(self):
        self.processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )
        model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa", torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):
        # prepare image + question
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
        text = "How many cats are there?"
        # prepare inputs
        encoding = self.processor(image, text, return_tensors="pt")
        encoding["pixel_values"] = encoding["pixel_values"].to(torch.bfloat16)
        return encoding


VILT_VARIANTS = [
    ModelMetadata(model_name="ViLT", relative_atol=0.02, model_group="red")
]


@pytest.mark.parametrize("model_info", VILT_VARIANTS, ids=lambda x: x.model_name)
@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
def test_vilt(record_property, model_info, mode, execute_mode):
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.op_by_op_backend = model_info.op_by_op_backend
    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
        cc.compile_depth = execute_mode
    else:
        cc.compile_depth = model_info.compile_depth

    tester = ThisTester(
        model_name=model_info.model_name,
        model_info=model_info,
        mode=mode,
        relative_atol=model_info.relative_atol,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_info.model_group,
    )
    results = tester.test_model()
    if mode == "eval":
        logits = results.logits
        idx = logits.argmax(-1).item()
        print("Predicted answer:", tester.framework_model.config.id2label[idx])

    tester.finalize()
