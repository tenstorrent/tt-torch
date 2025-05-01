# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/google/vit-base-patch16-224

from transformers import ViTImageProcessor, ViTForImageClassification
import requests
from PIL import Image
import pytest
import onnx
import torch
from tests.utils import OnnxModelTester
import os
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(OnnxModelTester):
    def _load_model(self):
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        m = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

        # Save the label mapping for eval
        self.id2label = m.config.id2label

        torch.onnx.export(m, self._load_torch_inputs(), f"{self.model_name}.onnx")
        self.model = onnx.load(f"{self.model_name}.onnx")
        os.remove(f"{self.model_name}.onnx")
        return self.model

    def _load_torch_inputs(self):
        # Load image
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        # Prepare input
        input = self.processor(images=image, return_tensors="pt")

        # Return just the tensor as tuple, not BatchFeature object
        return (input["pixel_values"],)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, None],
    ids=["op_by_op_stablehlo", "full"],
)
def test_vit_onnx(record_property, mode, op_by_op):
    model_name = "ViT"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        relative_atol=0.01,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group="red",
        assert_pcc=True,
        assert_atol=False,
    )

    results = tester.test_model()
    if mode == "eval":
        # Get the predicted class index
        logits = results[0]
        predicted_class_idx = logits.argmax(-1).item()
        print(
            "Predicted class:",
            tester.id2label[predicted_class_idx],
        )

    tester.finalize()
