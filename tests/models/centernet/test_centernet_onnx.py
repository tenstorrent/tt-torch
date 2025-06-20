# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from tests.utils import OnnxModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.centernet.pytorch import ModelLoader


class ThisTester(OnnxModelTester):
    def _load_model(self):
        """
        The model is from https://github.com/xingyizhou/CenterNet
        """
        # Model
        return ModelLoader.load_model()

    def _load_torch_inputs(self):
        # Images
        return ModelLoader.load_inputs()


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, None],
    ids=["op_by_op_stablehlo", "full"],
)
def test_centernet_onnx(record_property, mode, op_by_op):
    model_name = "CENTERNET_ONNX"
    model_group = "red"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    if op_by_op is not None:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        cc.op_by_op_backend = op_by_op

    # TODO Enable PCC/ATOL/Checking - https://github.com/tenstorrent/tt-torch/issues/976
    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
    )
    results = tester.test_model()

    if mode == "eval":
        # Results
        print(results)

    tester.finalize()
