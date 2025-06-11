# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file


class ThisTester(ModelTester):
    def _load_model(self):
        # Load the ResNet-50 model with updated API
        self.weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=self.weights)
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        # Define a transformation to preprocess the input image using the weights transforms
        preprocess = self.weights.transforms()

        # Load and preprocess the image
        # Local cache of http://images.cocodataset.org/val2017/000000039769.jpg
        image_file = get_file("test_images/coco_two_cats_000000039769_640x480.jpg")
        image = Image.open(str(image_file))
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0)
        batch_t = batch_t.to(torch.bfloat16)
        return batch_t


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_resnet(record_property, mode, op_by_op, data_parallel_mode):
    if mode == "train":
        pytest.skip()
    model_name = "ResNet50"

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
        required_atol=0.03,
        required_pcc=0.98,
        compiler_config=cc,
        assert_pcc=True,
        assert_atol=False,
        record_property_handle=record_property,
        data_parallel_mode=data_parallel_mode,
    )

    results = tester.test_model()

    def print_result(result):
        _, indices = torch.topk(result, 5)
        print(f"Top 5 predictions: {indices[0].tolist()}")

    if mode == "eval":
        ModelTester.print_outputs(results, data_parallel_mode, print_result)

    tester.finalize()


# Empty property record_property
def empty_record_property(a, b):
    pass


# Run pytorch implementation
if __name__ == "__main__":
    test_resnet(empty_record_property)
