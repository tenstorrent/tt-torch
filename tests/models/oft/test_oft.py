# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    ModelMetadata,
)
from third_party.tt_forge_models.oft.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return ModelLoader.load_model()

    def _load_inputs(self):
        return ModelLoader.load_inputs()


OFT_VARIANTS = [
    ModelMetadata(
        model_name="OFT",
        model_group="red",
        assert_atol=False,
        compile_depth=CompileDepth.TTNN_IR
    )
]


@pytest.mark.parametrize("model_info", OFT_VARIANTS, ids=lambda x: x.model_name)
@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
def test_oft(record_property, model_info, mode, execute_mode):
    if mode == "train":
        pytest.skip()

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    cc.op_by_op_backend = model_info.op_by_op_backend
    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
        cc.compile_depth = execute_mode
    else:
        cc.compile_depth = model_info.compile_depth

    skip_full_eval_test(
        record_property,
        cc,
        model_info.model_name,
        bringup_status="FAILED_RUNTIME",
        reason="Out of Memory: Not enough space to allocate 2902982656 B DRAM buffer across 12 banks - https://github.com/tenstorrent/tt-torch/issues/727",
        model_group=model_info.model_group,
    )

    tester = ThisTester(
        model_name=model_info.model_name,
        model_info=model_info,
        mode=mode,
        compiler_config=cc,
        assert_atol=model_info.assert_atol,
        record_property_handle=record_property,
        model_group=model_info.model_group,
    )

    results = tester.test_model()
    tester.finalize()
