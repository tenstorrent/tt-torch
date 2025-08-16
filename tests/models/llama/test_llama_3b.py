# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.llama.causal_lm.pytorch import ModelLoader


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
def test_llama_3b(record_property, mode, op_by_op):
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

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
        compiler_config=cc,
        assert_atol=False,
        # FIXME - PCC check should consider attention_mask: https://github.com/tenstorrent/tt-torch/issues/1176
        assert_pcc=False,
        required_pcc=0.96,
        record_property_handle=record_property,
    )
    results = tester.test_model()
    tester.finalize()
