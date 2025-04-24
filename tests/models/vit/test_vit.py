# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/google/vit-base-patch16-224

from transformers import ViTImageProcessor, ViTForImageClassification
import requests
from PIL import Image
import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224", torch_dtype=torch.bfloat16
        )
        m = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", torch_dtype=torch.bfloat16
        )
        return m

    def _load_inputs(self):
        # Load image
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        # Prepare input
        input = self.processor(images=image, return_tensors="pt")
        input["pixel_values"] = input["pixel_values"].to(torch.bfloat16)
        return input


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_vit(record_property, mode, op_by_op):
    model_name = "ViT"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    # TODO Enable checking - https://github.com/tenstorrent/tt-torch/issues/553
    tester = ThisTester(
        model_name,
        mode,
        relative_atol=0.01,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group="red",
        assert_pcc=True,
        assert_atol=True,
    )

    results = tester.test_model()
    if mode == "eval":
        # Get the predicted class index
        logits = results.logits
        predicted_class_idx = logits.argmax(-1).item()
        print(
            "Predicted class:",
            tester.framework_model.config.id2label[predicted_class_idx],
        )

    tester.finalize()
