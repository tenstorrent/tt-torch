# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from io import BytesIO
import requests
import json
from PIL import Image
import torch
from torchvision import transforms

class ThisTester(ModelTester):
    def _load_model(self):
        model = EfficientNet.from_pretrained('efficientnet-b0')
        return model.eval()

    def _load_inputs(self):
        # Fetch image from URL
        url = "https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/examples/simple/img.jpg"
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        img = Image.open(BytesIO(response.content))

        # Apply transformations
        tfms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return tfms(img).unsqueeze(0)

@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_EfficientNet(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "EfficientNet"
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO
    
    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group="red",
    )
    with torch.no_grad():
        results = tester.test_model()
    if mode == "eval":
        print("eval")
        # Fetch labels_map from URL
        labels_url = "https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/examples/simple/labels_map.txt"
        response = requests.get(labels_url)
        response.raise_for_status()  # Ensure the request was successful
        labels_map = json.loads(response.text)
        labels_map = [labels_map[str(i)] for i in range(1000)]

        print('-----')
        for idx in torch.topk(results, k=5).indices.squeeze(0).tolist():
            prob = torch.softmax(results, dim=1)[0, idx].item()
            print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
    tester.finalize()

# model = EfficientNet.from_pretrained('efficientnet-b0')

# # Preprocess image
# tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
# img = tfms(Image.open('img.jpg')).unsqueeze(0)
# print(img.shape) # torch.Size([1, 3, 224, 224])

# # Load ImageNet class names
# labels_map = json.load(open('labels_map.txt'))
# labels_map = [labels_map[str(i)] for i in range(1000)]

# # Classify
# model.eval()
# with torch.no_grad():
#     outputs = model(img)

# # Print predictions
# print('-----')
# for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
#     prob = torch.softmax(outputs, dim=1)[0, idx].item()
#     print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))