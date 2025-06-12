# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/facebook/deit-base-patch16-224

from transformers import AutoFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file


class ThisTester(ModelTester):
    def _load_model(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        model = ViTForImageClassification.from_pretrained(self.model_name)
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
        inputs = self.feature_extractor(images=image, return_tensors="pt")
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
    [
        pytest.param(
            "train",
            marks=pytest.mark.compilation_xfail,
        ),
        "eval",
    ],
)
@pytest.mark.parametrize("model_name", ["facebook/deit-base-patch16-224"])
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_deit(record_property, model_name, mode, op_by_op, data_parallel_mode):
    if mode == "train":
        pytest.skip()

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        required_pcc=0.97,
        relative_atol=0.015,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=True,
        assert_atol=False,
        data_parallel_mode=data_parallel_mode,
    )
    results = tester.test_model()

    def print_result(result):
        logits = result.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        print(
            "Predicted class:",
            tester.framework_model.config.id2label[predicted_class_idx],
        )

    if mode == "eval":
        ModelTester.print_outputs(results, data_parallel_mode, print_result)

    tester.finalize()
