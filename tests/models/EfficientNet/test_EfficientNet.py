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
from tt_torch.tools.utils import CompilerConfig, ModelMetadata, CompileDepth, OpByOpBackend


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
        return tfms(img).unsqueeze(0)

# Metadata for EfficientNet models
EFFICIENTNET_VARIANTS = [
    ModelMetadata(model_name="efficientnet-b0", model_group="red"),
    ModelMetadata(model_name="efficientnet-b1", assert_pcc=True),
    ModelMetadata(model_name="efficientnet-b2"),
    ModelMetadata(model_name="efficientnet-b3"),
    ModelMetadata(model_name="efficientnet-b4"),
    ModelMetadata(model_name="efficientnet-b5"),
    ModelMetadata(model_name="efficientnet-b6"),
    ModelMetadata(model_name="efficientnet-b7"),
]

@pytest.mark.parametrize("model_info", EFFICIENTNET_VARIANTS, ids=lambda x: x.model_name)
@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
     "execute_mode",
     [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
     ids=["op_by_op","full"],
)
def test_EfficientNet(record_property, model_info, mode, execute_mode):
    if mode == "train":
        pytest.skip()

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    # check if OpByOp
    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
        cc.compile_depth = execute_mode
        cc.op_by_op_backend = model_info.op_by_op_backend
    # applying overrides from model_metadata if EXECUTE
    else:
        cc.compile_depth = model_info.compile_depth
        cc.op_by_op_backend = model_info.op_by_op_backend
    
    required_pcc = (
        0.98
    )

    tester = ThisTester(
        model_name=model_info.model_name,
        model_info=model_info,
        mode=mode,
        required_pcc=required_pcc,
        assert_pcc=model_info.assert_pcc,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_info.model_group,
    )

    results = tester.test_model()

    if mode == "eval":
        print("eval")
        # Fetch labels_map from URL
        labels_map = json.load(open("tests/models/EfficientNet/labels_map.txt"))
        labels_map = [labels_map[str(i)] for i in range(1000)]

        print("-----")
        print(f"Model: {model_info.model_name}")
        for idx in torch.topk(results, k=5).indices.squeeze(0).tolist():
            prob = torch.softmax(results, dim=1)[0, idx].item()
            print("{label:<75} ({p:.2f}%)".format(label=labels_map[idx], p=prob * 100))
    tester.finalize()