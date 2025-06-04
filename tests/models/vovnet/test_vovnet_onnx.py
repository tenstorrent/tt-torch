# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import onnx
import requests
from PIL import Image
import os

import pytest
from tests.utils import OnnxModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend

dependencies = ["timm==1.0.9"]


class ThisTester(OnnxModelTester):
    def _load_model(self):
        import timm

        # Load the VovNet from timm
        self.torch_model = timm.create_model(
            "ese_vovnet19b_dw.ra_in1k", pretrained=True
        )
        model = self.torch_model.eval()

        # Export to ONNX
        torch.onnx.export(model, self._load_torch_inputs(), f"{self.model_name}.onnx")
        model = onnx.load(f"{self.model_name}.onnx")
        os.remove(f"{self.model_name}.onnx")
        return model

    def _load_torch_inputs(self):
        import timm

        # Get model specific transforms
        data_config = timm.data.resolve_model_data_config(self.torch_model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        # Load and preprocess the image
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        image = Image.open(requests.get(url, stream=True).raw)
        img_t = transforms(image)
        batch_t = torch.unsqueeze(img_t, 0)
        return (batch_t,)


@pytest.mark.usefixtures("manage_dependencies")
@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, None],
    ids=["op_by_op_stablehlo", "full"],
)
def test_vovnet_onnx(record_property, mode, op_by_op):
    model_name = "vovnet"
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
        top5_probabilities, top5_class_indices = torch.topk(
            results[0].softmax(dim=0) * 100, k=5
        )
        print(f"Top 5 predictions: {top5_class_indices.tolist()}")

    tester.finalize()
