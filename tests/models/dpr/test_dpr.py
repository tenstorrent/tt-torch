# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/facebook/dpr-reader-single-nq-base

import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.dpr.reader.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return ModelLoader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return ModelLoader.load_inputs(dtype_override=torch.bfloat16)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_dpr(record_property, mode, op_by_op, data_parallel_mode):
    model_name = "DPR"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=True,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        data_parallel_mode=data_parallel_mode,
    )
    results = tester.test_model()

    def print_result(result):
        # start_logits = result.start_logits
        # end_logits = result.end_logits
        # relevance_logits = result.relevance_logits
        print(result)

    if mode == "eval":
        ModelTester.print_outputs(results, data_parallel_mode, print_result)

    tester.finalize()
