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
from third_party.tt_forge_models.qwen.casual_lm.pytorch import ModelLoader


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
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    # TODO: Remove this once PCC ATOL is fixed on blackhole runners - https://github.com/tenstorrent/tt-torch/issues/1003
    assert_pcc = tt_mlir.get_arch() != tt_mlir.Arch.BLACKHOLE

    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)

    tester = ThisTester(
        model_info.name,
        mode,
        loader=loader,
        model_info=model_info,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=assert_pcc,
        assert_atol=False,
        run_generate=False,
        required_pcc=0.85,
    )

    results = tester.test_model()

    if mode == "eval":
        gen_text = loader.decode_output(results, dtype_override=torch.bfloat16)

        print(
            f"Model: {model_info.name} | Input: {loader.text} | Decoded Text: {gen_text}"
        )

    tester.finalize()
