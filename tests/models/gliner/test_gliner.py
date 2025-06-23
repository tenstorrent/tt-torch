# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.gliner.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return ModelLoader.load_model()

    def _load_inputs(self):
        return ModelLoader.load_inputs()


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_gliner(record_property, mode, op_by_op):
    model_name = "gliner-v2"
    model_group = "red"

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
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
    )

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_TTMLIR_COMPILATION",
        reason="moreh_softmax_device_operation.cpp Inputs must be of bfloat16 or bfloat8_b type - https://github.com/tenstorrent/tt-torch/issues/732",
        model_group=model_group,
    )

    entities = tester.test_model()
    if mode == "eval":
        for entity in entities:
            print(entity["text"], "=>", entity["label"])
    tester.finalize()
