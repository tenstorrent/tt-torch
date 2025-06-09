# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from tests.utils import ModelTester  # for PyTorch Tests
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from transformers.image_utils import load_image
from transformers import DFineForObjectDetection, AutoImageProcessor


class ThisTester(ModelTester):
    def _load_model(self):
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        model = DFineForObjectDetection.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.image = load_image(url)
        inputs = self.processor(images=self.image, return_tensors="pt")
        inputs = inputs.to(torch.bfloat16)
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "ustc-community/dfine-nano-coco" "ustc-community/dfine-small-coco",
        "ustc-community/dfine-medium-coco",
        "ustc-community/dfine-large-coco",
        "ustc-community/dfine-xlarge-coco",
    ],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_d_fine(record_property, model_name, mode, op_by_op):

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()

    if mode == "eval":
        results = tester.processor.post_process_object_detection(
            results,
            target_sizes=[(tester.image.height, tester.image.width)],
            threshold=0.5,
        )
        for result in results:
            for score, label_id, box in zip(
                result["scores"], result["labels"], result["boxes"]
            ):
                score, label = score.item(), label_id.item()
                box = [round(i, 2) for i in box.tolist()]
                print(f"{tester.model.config.id2label[label]}: {score:.2f} {box}")

    tester.finalize()
