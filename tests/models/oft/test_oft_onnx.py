# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import onnx
import pytest
import os
from tests.utils import OnnxModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.oft import OFTLoader


class ThisTester(OnnxModelTester):
    def _load_model(self):
        self.grid_res = 0.5
        model = OFTLoader.load_model(
            num_classes=1,
            frontend="resnet18",
            topdown_layers=8,
            grid_res=self.grid_res,
            grid_height=4.0,
        )
        torch.onnx.export(model, self._load_torch_inputs(), f"{self.model_name}.onnx")
        model = onnx.load(f"{self.model_name}.onnx")
        onnx.checker.check_model(model)
        os.remove(f"{self.model_name}.onnx")
        return model

    def _load_torch_inputs(self):
        return OFTLoader.load_inputs(grid_res=self.grid_res, grid_size=(80.0, 80.0))


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize("op_by_op", [True, False], ids=["op_by_op_stablehlo", "full"])
def test_oft_onnx(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "OFT"

    cc = CompilerConfig()

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
