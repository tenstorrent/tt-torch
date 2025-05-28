# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from io import BytesIO
import requests
import json
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, ModelMetadata


class ThisTester(ModelTester):
    def _load_model(self):
        model = EfficientNet.from_pretrained(self.model_name)
        return model.eval()

    def _load_inputs(self):
        # Fetch image from URL
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        img = Image.open(BytesIO(response.content))

        # Apply transformations
        tfms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return tfms(img).unsqueeze(0)




EFFICIENTNET_METADATA= {
    "efficientnet-b0": ModelMetadata(model_name="efficientnet-b0", compile_depth=CompileDepth.STABLEHLO),
    "efficientnet-b1": ModelMetadata(model_name="efficientnet-b1", compile_depth=CompileDepth.STABLEHLO),
    "efficientnet-b2": ModelMetadata(model_name="efficientnet-b2", compile_depth=CompileDepth.STABLEHLO),
    "efficientnet-b3": ModelMetadata(model_name="efficientnet-b3", compile_depth=CompileDepth.STABLEHLO),
    "efficientnet-b4": ModelMetadata(model_name="efficientnet-b4", compile_depth=CompileDepth.STABLEHLO),
    "efficientnet-b5": ModelMetadata(model_name="efficientnet-b5", compile_depth=CompileDepth.STABLEHLO),
    "efficientnet-b6": ModelMetadata(model_name="efficientnet-b6", compile_depth=CompileDepth.STABLEHLO),
    "efficientnet-b7": ModelMetadata(model_name="efficientnet-b7", compile_depth=CompileDepth.STABLEHLO),
}

@pytest.mark.model_metadata(model_metadata=EFFICIENTNET_METADATA)
@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize("model_name", EFFICIENTNET_METADATA.keys())
@pytest.mark.parametrize(
     "execute_mode",
     [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
     ids=["op_by_op","full"],
)
def test_EfficientNet(record_property, model_name, mode, execute_mode, model_metadata_fixture):
    if mode == "train":
        pytest.skip()

    model_group = "red" if model_name == "efficientnet-b0" else "generality"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    
    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
        cc.compile_depth = execute_mode

    model_metadata_to_run = model_metadata_fixture.get(model_name)

    # applying overrides from model_metadata_fixture
    if model_metadata_to_run:
        if model_metadata_to_run.compile_depth is not None:
            cc.compile_depth = model_metadata_to_run.compile_depth
        if model_metadata_to_run.op_by_op_backend is not None:
            cc.op_by_op_backend = model_metadata_to_run.op_by_op_backend


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
        for idx in torch.topk(results, k=5).indices.squeeze(0).tolist():
            prob = torch.softmax(results, dim=1)[0, idx].item()
            print("{label:<75} ({p:.2f}%)".format(label=labels_map[idx], p=prob * 100))
    tester.finalize()
