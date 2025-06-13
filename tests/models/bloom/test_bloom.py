# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

# Load model directly
from third_party.tt_forge_models.bloom.pytorch import ModelLoader
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


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
def test_bloom(record_property, mode, op_by_op):
    model_name = "Bloom"

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
        relative_atol=0.01,
        assert_pcc=True,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()

    if mode == "eval":
        decoded_output = ModelLoader.decode_output(results)

        print(f"model_name: {model_name}")
        for i, (inp, out) in enumerate(zip(ModelLoader.test_input, decoded_output)):
            print(f"input {i}: {inp}\noutput {i}: {out}\n")

    tester.finalize()
