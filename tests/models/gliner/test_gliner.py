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
        return self.loader.load_model()

    def _load_inputs(self):
        return self.loader.load_inputs()


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
    cc = CompilerConfig()
    cc.enable_consteval = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)

    tester = ThisTester(
        model_info.name,
        mode,
        loader=loader,
        model_info=model_info,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )

    skip_full_eval_test(
        record_property,
        cc,
        model_info.name,
        bringup_status="FAILED_TTMLIR_COMPILATION",
        reason="moreh_softmax_device_operation.cpp Inputs must be of bfloat16 or bfloat8_b type - https://github.com/tenstorrent/tt-torch/issues/732",
        model_group=model_info.group,
    )

    entities = tester.test_model()
    if mode == "eval":
        for batch_entities in entities:
            for entity in batch_entities:
                print(entity["text"], "=>", entity["label"])
    tester.finalize()
