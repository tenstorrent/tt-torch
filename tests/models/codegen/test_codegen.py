# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/model_doc/codegen#usage-example

import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.codegen.pytorch import ModelLoader


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
def test_codegen(record_property, mode, op_by_op):
    model_name = "codegen"
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
        run_generate=False,
        assert_atol=False,
    )

    results = tester.test_model()

    if mode == "eval":
        logits = results.logits if hasattr(results, "logits") else results[0]
        token_ids = torch.argmax(logits, dim=-1)
        decoded = tester.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        print(decoded)

    tester.finalize()
