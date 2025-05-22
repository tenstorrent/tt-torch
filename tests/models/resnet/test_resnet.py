# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torchvision
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from tt_torch.tools.device_manager import DeviceManager


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
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_resnet(record_property, data_parallel_mode, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "ResNet18"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO
    devices = None
    if data_parallel_mode:
        parent, devices = DeviceManager.acquire_available_devices()
    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=True,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        data_parallel_mode=data_parallel_mode,
        devices=devices,
    )
    results = tester.test_model()
    if data_parallel_mode:
        DeviceManager.release_parent_device(parent, cleanup_sub_devices=True)
    tester.finalize()
