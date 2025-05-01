# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torchvision
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
import time


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
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_resnet(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "ResNet18"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.cache_preprocessed_constants = True

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
    )

    model = tester.compile_model(tester.get_framework_model(), tester.compiler_config)

    num_loops = 10
    with torch.no_grad():
        start_first = time.time()
        results = tester.run_model(model, tester.inputs)
        end_first = time.time()
        print(f"First iteration took {(end_first - start_first)} seconds")

        start_time = time.time()
        for _ in range(num_loops):
            results = tester.run_model(model, tester.inputs)
        end_time = time.time()

        print(f"Model: {model_name}")
        print(f"{num_loops} iterations took {(end_time - start_time)} seconds")
        print(f"Average iteration time: {(end_time - start_time) / num_loops} seconds")

    tester.finalize()
