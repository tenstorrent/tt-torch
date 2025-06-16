# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import onnx
from PIL import Image
import os

import pytest
from tests.utils import OnnxModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file
from torchvision import models


class ThisTester(OnnxModelTester):
    def _load_model(self):
        # Load the Swin V2 S model from torchvision
        self.weights = models.Swin_V2_S_Weights.DEFAULT
        self.torch_model = models.get_model("swin_v2_s", weights=self.weights).eval()

        # Export to ONNX with specific settings for Swin V2
        torch_inputs = self._load_torch_inputs()
        print(f"Input shape: {torch_inputs[0].shape}", flush=True)

        torch.onnx.export(
            self.torch_model, self._load_torch_inputs(), f"{self.model_name}.onnx"
        )
        model = onnx.load(f"{self.model_name}.onnx")
        os.remove(f"{self.model_name}.onnx")
        return model

    def _load_torch_inputs(self):
        # Get model specific transforms
        preprocess = self.weights.transforms()

        # Load and preprocess the image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0).to(torch.float32)
        return (batch_t,)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, None],
    ids=["op_by_op_stablehlo", "full"],
)
def test_swin_v2_onnx(record_property, mode, op_by_op):
    model_name = "swin_v2_s"
    model_group = "red"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    if op_by_op is not None:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        cc.op_by_op_backend = op_by_op

    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=True,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
    )
    results = tester.test_model()

    if mode == "eval":
        # Print the top 5 predictions
        _, indices = torch.topk(results[0], 5)
        print(f"Model: {model_name} | Top 5 predictions: {indices[0].tolist()}")

    tester.finalize()
