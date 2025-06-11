# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/google/vit-base-patch16-224

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import pytest
import onnx
import torch
from tests.utils import OnnxModelTester
import os
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file


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
        # Local cache of http://images.cocodataset.org/val2017/000000039769.jpg
        image_file = get_file("test_images/coco_two_cats_000000039769_640x480.jpg")
        image = Image.open(str(image_file))
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
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_vit_onnx(record_property, mode, op_by_op, data_parallel_mode):
    model_name = "ViT"

    if data_parallel_mode:
        pytest.skip("Data parallel mode not supported for onnx models yet")

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
