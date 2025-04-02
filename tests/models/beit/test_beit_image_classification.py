# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import BeitImageProcessor, BeitForImageClassification
from PIL import Image
import requests
import pytest
import torch
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        model = BeitForImageClassification.from_pretrained(self.model_name)
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        processor = BeitImageProcessor.from_pretrained(self.model_name)
        inputs = processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs

    def set_inputs_train(self, inputs):
        inputs["pixel_values"].requires_grad_(True)
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
    "model_name",
    ["microsoft/beit-base-patch16-224", "microsoft/beit-large-patch16-224"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_beit_image_classification(record_property, model_name, mode, op_by_op):
    if mode == "train":
        pytest.skip()

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    required_atol = 0.032 if model_name == "microsoft/beit-base-patch16-224" else 0.065
    tester = ThisTester(
        model_name,
        mode,
        required_atol=required_atol,
        compiler_config=cc,
        record_property_handle=record_property,
        # TODO Enable checking - https://github.com/tenstorrent/tt-torch/issues/550
        assert_pcc=False,
        assert_atol=False,
    )
    results = tester.test_model()

    if mode == "eval":
        logits = results.logits

        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        print(
            "Predicted class:",
            tester.framework_model.config.id2label[predicted_class_idx],
        )

    tester.finalize()
