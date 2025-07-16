# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/dandelin/vilt-b32-finetuned-vqa

from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.vilt.pytorch.loader import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs()


# class ThisTester(ModelTester):
#     def _load_model(self):
#         self.processor = ViltProcessor.from_pretrained(
#             "dandelin/vilt-b32-finetuned-vqa"
#         )
#         model = ViltForQuestionAnswering.from_pretrained(
#             "dandelin/vilt-b32-finetuned-vqa", torch_dtype=torch.bfloat16
#         )
#         return model

#     def _load_inputs(self):
#         # prepare image + question
#         image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
#         image = Image.open(str(image_file))
#         text = "How many cats are there?"
#         # prepare inputs
#         encoding = self.processor(image, text, return_tensors="pt")
#         encoding["pixel_values"] = encoding["pixel_values"].to(torch.bfloat16)
#         return encoding


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_vilt(record_property, mode, op_by_op):
    model_name = "ViLT"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)

    tester = ThisTester(
        model_info.name,
        mode,
        loader=loader,
        model_info=model_info,
        compiler_config=cc,
        assert_atol=False,
        record_property_handle=record_property,
    )
    results = tester.test_model()
    if mode == "eval":
        if isinstance(results, tuple):
            logits = results[0]  # Assuming logits are the first element
        else:
            logits = results.logits
        idx = logits.argmax(-1).item()
        print("Predicted answer:", tester.framework_model.config.id2label[idx])

    tester.finalize()
