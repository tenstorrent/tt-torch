# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file


class ThisTester(ModelTester):
    def _load_model(self):
        model = EfficientNet.from_pretrained(self.model_name)
        return model.eval()

    def _load_inputs(self):
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
        return torch.stack([tfms(img)] * 2)


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
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
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_EfficientNet(record_property, model_name, mode, op_by_op):
    if mode == "train":
        pytest.skip()

    model_group = "red" if model_name == "efficientnet-b0" else "generality"

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

    required_pcc = (
        0.98
        if model_name
        in [
            "efficientnet-b1",
        ]
        else 0.99
    )

    tester = ThisTester(
        model_name,
        mode,
        required_pcc=required_pcc,
        assert_pcc=True,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
    )

    results = tester.test_model()

    if mode == "eval":
        print("eval")
        # Fetch labels_map from URL
        labels_map = json.load(open("tests/models/EfficientNet/labels_map.txt"))
        labels_map = [labels_map[str(i)] for i in range(1000)]

        print("-----")
        print(f"Model: {model_name}")
        for i in range(results.shape[0]):
            result_i = results[i]
            print(f"Output {i+1}:")
            for idx in torch.topk(result_i, k=5).indices.squeeze(0).tolist():
                prob = torch.softmax(result_i, dim=0)[idx].item()
                print(
                    "{label:<75} ({p:.2f}%)".format(label=labels_map[idx], p=prob * 100)
                )
    tester.finalize()
