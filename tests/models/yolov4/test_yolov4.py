# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from third_party.tt_forge_models.yolov4.pytorch import ModelLoader
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
import tt_mlir


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs(dtype_override=torch.bfloat16)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_yolov4(record_property, mode, op_by_op):
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)

    # TODO: Remove this once PCC ATOL is fixed for yolov4 on blackhole runners - https://github.com/tenstorrent/tt-torch/issues/1003
    assert_pcc = tt_mlir.get_arch() != tt_mlir.Arch.BLACKHOLE
    assert_atol = tt_mlir.get_arch() != tt_mlir.Arch.BLACKHOLE

    tester = ThisTester(
        model_info.name,
        mode,
        loader=loader,
        model_info=model_info,
        compiler_config=cc,
        assert_pcc=assert_pcc,
        assert_atol=assert_atol,
        record_property_handle=record_property,
        model_group="red",
    )
    with torch.no_grad():
        tester.test_model()
    tester.finalize()
