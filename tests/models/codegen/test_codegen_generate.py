# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/model_doc/codegen#usage-example

from transformers import AutoModelForCausalLM, AutoTokenizer
import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        model = ModelLoader.load_model(dtype_override=torch.bfloat16)
        self.tokenizer = ModelLoader.tokenizer
        return model

    def _load_inputs(self):
        return ModelLoader.load_inputs(dtype_override=torch.bfloat16, batch_size=2)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_codegen_generate(record_property, mode, op_by_op):
    model_name = "codegen_generate"
    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        is_token_output=True,
        run_generate=True,  # run model.generate(**inputs)
        assert_atol=False,
    )

    results = tester.test_model(
        assert_eval_token_mismatch=False
    )  # don't validate token output

    if mode == "eval":
        decoded_outputs = ModelLoader.decode_outputs(results)
        for i, text in enumerate(decoded_outputs):
            print(f"Output {i}: {text}")

    tester.finalize()
