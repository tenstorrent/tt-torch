# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from tests.utils import OnnxModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.oft.onnx import ModelLoader


class ThisTester(OnnxModelTester):
    def _load_model(self):
        return self.loader.load_model()

    def _load_torch_inputs(self):
        return self.loader.load_inputs()


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize("op_by_op", [True, False], ids=["op_by_op_stablehlo", "full"])
def test_oft_onnx(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()

    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)

    cc = CompilerConfig()

    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_info.name,
        mode,
        loader=loader,
        model_info=model_info,
        compiler_config=cc,
        assert_atol=False,
        record_property_handle=record_property,
    )

    results = tester.test_model()
    tester.finalize()
