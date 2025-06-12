# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import onnx
import os
import numpy as np
import json
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from tests.utils import OnnxModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file


class ThisTester(OnnxModelTester):
    def _load_model(self):
        model = EfficientNet.from_pretrained(self.model_name)
        model.set_swish(memory_efficient=False)
        model.eval()
        torch.onnx.export(model, self._load_torch_inputs(), f"{self.model_name}.onnx")
        self.model = onnx.load(f"{self.model_name}.onnx")
        os.remove(f"{self.model_name}.onnx")
        return self.model

    def _load_torch_inputs(self):
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        img = Image.open(str(image_file))

        # Apply transformations
        tfms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return tfms(img).unsqueeze(0)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "efficientnet-b4",
        "efficientnet-b5",
        "efficientnet-b6",
        "efficientnet-b7",
    ],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, None],
    ids=["op_by_op_stablehlo", "full"],
)
def test_EfficientNet_onnx(record_property, model_name, mode, op_by_op):
    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    model_group = "red" if model_name == "efficientnet-b0" else "generality"

    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
    )

    results = tester.test_model()
    if mode == "eval":
        print("eval")
        labels_map = json.load(open("tests/models/EfficientNet/labels_map.txt"))
        labels_map = [labels_map[str(i)] for i in range(1000)]

        print("-----")
        print(f"Model: {model_name}")
        for idx in torch.topk(results[0], k=5).indices.squeeze(0).tolist():
            prob = torch.softmax(results[0], dim=1)[0, idx].item()
            print("{label:<75} ({p:.2f}%)".format(label=labels_map[idx], p=prob * 100))
    tester.finalize()
