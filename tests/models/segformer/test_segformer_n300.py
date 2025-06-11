# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file


class ThisTester(ModelTester):
    def _load_model(self):
        self.processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        # Local cache of http://images.cocodataset.org/val2017/000000039769.jpg
        image_file = get_file("test_images/coco_two_cats_000000039769_640x480.jpg")
        image = Image.open(str(image_file))
        inputs = self.processor(images=([image] * 8), return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs

    def set_inputs_train(self, inputs):
        inputs["pixel_values"] = inputs["pixel_values"].requires_grad_(True)
        return inputs

    def append_fake_loss_function(self, outputs):
        return torch.mean(outputs.logits)

    def get_results_train(self, model, inputs, outputs):
        return inputs["pixel_values"].grad


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_segformer(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "SegFormer"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.automatic_parallelization = True
    cc.mesh_shape = [1, 2]
    cc.dump_info = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        relative_atol=0.01,
        compiler_config=cc,
        # TODO - Enable checking - https://github.com/tenstorrent/tt-torch/issues/527
        assert_atol=False,
        record_property_handle=record_property,
        model_group="red",
    )
    results = tester.test_model()
    if mode == "eval":
        logits = results.logits  # shape (batch_size, num_labels, height/4, width/4)

    tester.finalize()
