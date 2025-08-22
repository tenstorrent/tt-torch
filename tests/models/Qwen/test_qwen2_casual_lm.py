# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/Qwen/Qwen2.5-1.5B
import torch
import pytest

# Load model directly
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
import tt_mlir
from third_party.tt_forge_models.qwen_2_5.casual_lm.pytorch import (
    ModelLoader,
    ModelVariant,
)


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs(dtype_override=torch.bfloat16)


@pytest.mark.parametrize(
    "mode",
    ["eval", "train"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_qwen2_casual_lm(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    variant = ModelVariant.QWEN_2_5_1_5B
    loader = ModelLoader(variant=variant)
    model_info = loader.get_model_info(variant=variant)

    tester = ThisTester(
        model_info.name,
        mode,
        loader=loader,
        model_info=model_info,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=True,
        assert_atol=False,
        run_generate=False,
        required_pcc=0.93,
    )

    results = tester.test_model()

    # TODO - decode_output() recently removed from ModelLoader. Consider bringing it back.
    # if mode == "eval":
    #     gen_text = loader.decode_output(results, dtype_override=torch.bfloat16)

    #     print(
    #         f"Model: {model_info.name} | Input: {loader.text} | Decoded Text: {gen_text}"
    #     )

    tester.finalize()
