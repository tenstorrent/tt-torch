# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# From: https://huggingface.co/alibaba-damo/mgp-str-base

import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.mgp_str_base.pytorch.loader import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs(dtype_override=torch.bfloat16)


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_mgp_str_base(record_property, mode, op_by_op, data_parallel_mode):

    # data parallel mode for this model currently removed from nightly tests

    if mode == "train":
        pytest.skip()

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.dump_info = True
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    # TODO Enable checking - https://github.com/tenstorrent/tt-torch/issues/552
    disable_checking = True

    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)

    tester = ThisTester(
        model_info.name,
        mode,
        loader=loader,
        model_info=model_info,
        relative_atol=0.02,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=True,
        assert_atol=False
        if disable_checking
        else True,  # ATOL checking issues - No model legitimately checks ATOL, issue #690
        data_parallel_mode=data_parallel_mode,
    )
    # TODO: This model test is still failing even with the to_host fix.
    results = tester.test_model()

    def print_result(result):
        logits = result.logits
        generated_text = tester.processor.batch_decode(logits)["generated_text"]
        print(f"Generated text: '{generated_text}'")
        assert generated_text[0] == "ticket"

    if mode == "eval" and not disable_checking:
        ModelTester.print_outputs(results, data_parallel_mode, print_result)

    tester.finalize()
