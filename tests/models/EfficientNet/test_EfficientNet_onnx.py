# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import onnx
import os
import numpy as np
from io import BytesIO
import requests
import json

# import onnxruntime
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from tests.utils import OnnxModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(OnnxModelTester):
    def _load_model(self):
        model = EfficientNet.from_pretrained("efficientnet-b1")
        model.set_swish(memory_efficient=False)
        model.eval()
        torch.onnx.export(model, self._load_torch_inputs(), f"{self.model_name}.onnx")
        self.model = onnx.load(f"{self.model_name}.onnx")
        os.remove(f"{self.model_name}.onnx")
        return self.model

    def _load_torch_inputs(self):
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


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, None],
    ids=["op_by_op_stablehlo", "full"],
)
def test_EfficientNet_onnx(record_property, mode, op_by_op):
    model_name = "EfficientNet_onnx"
    cc = CompilerConfig()
    cc.compile_depth = CompileDepth.STABLEHLO
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
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
    breakpoint()
    # onnx_model_path = f"{tester.model_name}.onnx"
    # onnx.save(tester.model, onnx_model_path)  # Save the loaded ONNX model
    # ort_session = onnxruntime.InferenceSession(onnx_model_path)

    # # Get input and output names
    # input_name = ort_session.get_inputs()[0].name
    # output_name = ort_session.get_outputs()[0].name

    # # Prepare the input
    # torch_input = tester._load_torch_inputs()
    # onnx_input = {input_name: torch_input.cpu().numpy()}

    # # Run the ONNX model
    # ort_outputs = ort_session.run([output_name], onnx_input)

    # # You can now assert on the output of the ONNX model
    # assert isinstance(ort_outputs, list)
    # assert len(ort_outputs) == 1
    # assert isinstance(ort_outputs[0], np.ndarray)
    # assert ort_outputs[0].shape[0] == 1  # Batch size should be 1
    # assert ort_outputs[0].shape[1] == 1000 # EfficientNet-b1 has 1000 output classes for ImageNet

    # # Clean up the saved ONNX model
    # os.remove(onnx_model_path)
    results = tester.test_model()
    if mode == "eval":
        print("eval")
        breakpoint()
        # Fetch labels_map from URL
        labels_map = json.load(open("tests/models/EfficientNet/labels_map.txt"))
        labels_map = [labels_map[str(i)] for i in range(1000)]

        print("-----")
        for idx in torch.topk(results, k=5).indices.squeeze(0).tolist():
            prob = torch.softmax(results, dim=1)[0, idx].item()
            print("{label:<75} ({p:.2f}%)".format(label=labels_map[idx], p=prob * 100))
    tester.finalize()
