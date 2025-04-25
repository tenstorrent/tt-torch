# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.oft import OFTLoader


class ThisTester(ModelTester):
    def _load_model(self):
        self.grid_res = 0.5
        model = OFTLoader.load_model(
            num_classes=1,
            frontend="resnet18",
            topdown_layers=8,
            grid_res=self.grid_res,
            grid_height=4.0,
        )
        return model

    def _load_inputs(self):
        return OFTLoader.load_inputs(grid_res=self.grid_res, grid_size=(80.0, 80.0))


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_oft(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "OFT"

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
        compiler_config=cc,
        assert_atol=False,
        record_property_handle=record_property,
        model_group="red",
    )

    results = tester.test_model()
    tester.finalize()
