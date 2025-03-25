# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

# Load model directly
from transformers import MistralConfig, MistralForCausalLM
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        config = MistralConfig()
        m = MistralForCausalLM(config)
        return m

    def _load_inputs(self):
        # Set up sample input with random values
        inputs = {}
        inputs["input_ids"] = torch.randint(1, 30001, (1, 12))
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_mistral_7b(record_property, mode, op_by_op):
    if op_by_op is None:
        pytest.skip("full-eval: Mistral-7B is too large to fit on a single device")

    model_name = "Mistral-7B"

    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=False,
        assert_pcc=False,
    )
    results = tester.test_model()

    tester.finalize()
