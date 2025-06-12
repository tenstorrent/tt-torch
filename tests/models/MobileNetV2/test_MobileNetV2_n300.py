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
        # Load the MobileNetV2 model with updated API
        self.weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=self.weights)
        return model.to(torch.bfloat16)

    def _load_inputs(self):
        # Define a transformation to preprocess the input image using the weights transforms
        preprocess = self.weights.transforms()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
        img_t = preprocess(image)
        batch_t = torch.stack([img_t] * 32)
        return batch_t.to(torch.bfloat16)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_MobileNetV2(record_property, mode, op_by_op):
    model_name = "MobileNetV2"
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
        required_pcc=0.98,
        assert_pcc=True,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group="red",
    )
    results = tester.test_model()
    if mode == "eval":
        # Print the top 5 predictions
        _, indices = torch.topk(results, 5)
        print(f"Top 5 predictions: {indices[0].tolist()}")

    tester.finalize()


# Empty property record_property
def empty_record_property(a, b):
    pass


# Main
if __name__ == "__main__":
    test_MobileNetV2(empty_record_property)
