# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import onnx
import requests
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

import pytest
from tests.utils import OnnxModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth


class ThisTester(OnnxModelTester):
    def _load_model(self):
        model = onnx.load("/localdev/achoudhury/customer_models/quantized_qdq/priorityA/bevdepth/bevdepth_ptq_qdq_dummy_v2_part1.onnx")
        return model


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize("op_by_op", [True, False], ids=["op_by_op", "full"])
def test_MobileNetV2(record_property, mode, op_by_op):
    model_name = "BevDepth"
    cc = CompilerConfig()
    cc.compile_depth = CompileDepth.STABLEHLO
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()
    if mode == "eval":
        # Print the top 5 predictions
        _, indices = torch.topk(results[0], 5)
        print(f"Top 5 predictions: {indices[0].tolist()}")

    tester.finalize()


# Empty property record_property
def empty_record_property(a, b):
    pass


# Main
if __name__ == "__main__":
    test_MobileNetV2(empty_record_property)
