# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from FlagEmbedding import BGEM3FlagModel


class ThisTester(ModelTester):
    def _load_model(self):
        model = BGEM3FlagModel("BAAI/bge-m3")
        self.model = model.model
        return self.model

    def _load_inputs(self):
        texts = [
            "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
            "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
        ]

        tokenized = self.model.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )

        forward_inputs = {
            "text_input": tokenized,
            "return_dense": True,
            "return_sparse": True,
            "return_colbert_vecs": True,
            "return_sparse_embedding": False,
        }

        return forward_inputs

    def _extract_outputs(self, output_object):
        if isinstance(output_object, dict):
            tensors = []
            for key, value in output_object.items():
                if isinstance(value, torch.Tensor):
                    tensors.append(value)

            if tensors:
                return tuple(tensors)
            else:
                raise ValueError(
                    f"No tensors found in output dictionary. Keys: {list(output_object.keys())}"
                )

            return super()._extract_outputs(output_object)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_bge_m3(record_property, mode, op_by_op):

    model_name = "BAAI/bge-m3"

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
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=True,
        assert_atol=False,
    )
    results = tester.test_model()
    tester.finalize()
