# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import BeitImageProcessor, BeitForImageClassification
from PIL import Image
import pytest
import torch
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file


class ThisTester(ModelTester):
    def _load_model(self):
        model = BeitForImageClassification.from_pretrained(self.model_name)
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
        processor = BeitImageProcessor.from_pretrained(self.model_name)
        images = [image] * 16  # Create a batch of 16
        inputs = processor(images=images, return_tensors="pt")
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
    cc.automatic_parallelization = True
    cc.mesh_shape = [1, 2]
    cc.dump_debug = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    required_atol = 0.032 if model_name == "microsoft/beit-base-patch16-224" else 0.065
    do_assert = True
    tester = ThisTester(
        model_name,
        mode,
        required_atol=required_atol,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=do_assert,
        assert_atol=False,
        # FIXME fails with tt-experimental - https://github.com/tenstorrent/tt-torch/issues/1105
        backend="tt",
    )
    results = tester.test_model()

    if mode == "eval":
        logits = results.logits

        # model predicts one of the 1000 ImageNet classes
        predicted_class_indices = logits.argmax(-1)
        for i, class_idx in enumerate(predicted_class_indices):
            print(
                f"Sample {i}: Predicted class: {tester.framework_model.config.id2label[class_idx.item()]}"
            )

    tester.finalize()
