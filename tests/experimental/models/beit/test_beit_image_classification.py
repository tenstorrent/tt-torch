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
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_beit_image_classification(
    record_property, model_name, mode, op_by_op, data_parallel_mode
):
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

def test_beit_image_classification_eager():
    model_name = "microsoft/beit-base-patch16-224"
    model = BeitForImageClassification.from_pretrained(model_name)
    model = model.to(torch.bfloat16).eval()

    processor = BeitImageProcessor.from_pretrained(model_name)

    image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(str(image_file))
    inputs = processor(images=image, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    breakpoint()
    cpu_results = model(**inputs)


    model = model.to("xla").eval()
    inputs = inputs.to("xla")

    #breakpoint()
    
    tt_results = model(**inputs).logits.to("cpu")

    
    cpu_logits = cpu_results.logits

    # model predicts one of the 1000 ImageNet classes
    tt_predicted_class_idx = tt_results.argmax(-1).item()
    cpu_predicted_class_idx = cpu_logits.argmax(-1).item()
    print(
        "Predicted class TT:",
        model.config.id2label[tt_predicted_class_idx],
    )

    print(
        "Predicted class CPU:",
        model.config.id2label[cpu_predicted_class_idx],
    )


