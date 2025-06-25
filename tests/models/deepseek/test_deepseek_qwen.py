# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.deepseek.qwen.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return ModelLoader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return ModelLoader.load_inputs(dtype_override=torch.bfloat16)


@pytest.mark.parametrize(
    "mode",
    ["eval", "train"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_deepseek_qwen(record_property, mode, op_by_op):
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    if mode == "train":
        pytest.skip()

    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    if op_by_op is None and cc.compile_depth == CompileDepth.EXECUTE:
        pytest.skip("Model is too large to fit on single device during execution.")

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        required_atol=0.5,
    )

    tester.test_model()
    tester.finalize()
