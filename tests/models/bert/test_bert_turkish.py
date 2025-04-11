# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
breakpoint()
import torch
import pytest

from sentence_transformers import SentenceTransformer
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend

breakpoint()


class ThisTester(ModelTester):
    def _load_model(self):
        model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
        return model.encode

    def _load_inputs(self):
        sentences = ["Bu örnek bir cümle", "Her cümle vektöre çevriliyor"]
        return sentences


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_bert_turkish(record_property, mode, op_by_op):
    model_name = "BERT_Turkish"

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
        relative_atol=0.012,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()

    tester.finalize()
