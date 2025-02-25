# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torchvision
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        model = torchvision.models.get_model("resnet18", pretrained=True)
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        inputs = torch.rand((1, 3, 224, 224), dtype=torch.bfloat16)
        inputs = inputs.to(torch.bfloat16)
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize("op_by_op", [True, False], ids=["op_by_op", "full"])
def test_resnet(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "ResNet18"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
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

    tester.finalize()
