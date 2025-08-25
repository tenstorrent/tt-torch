# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from transformers import AutoTokenizer, AutoModel

from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        model_name = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        return model

    def _load_inputs(self):
        sentences = ["Bu örnek bir cümle", "Her cümle vektöre çevriliyor"]
        self.input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        return self.input


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode",
    [False, True],
    ids=["single_device", "data_parallel"],
)
def test_bert_turkish(record_property, mode, op_by_op, data_parallel_mode):
    model_name = "BERT_Turkish"

    cc = CompilerConfig()
    cc.enable_consteval = True
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=True,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group="red",
        data_parallel_mode=data_parallel_mode,
    )
    with torch.no_grad():
        results = tester.test_model()

    if mode == "eval":

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[
                0
            ]  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        if data_parallel_mode:
            for i in range(len(results)):
                result = results[i]
                sentence_embeddings = mean_pooling(
                    result, tester.input["attention_mask"]
                )
                print(f"Device {i} | Sentence embeddings: {sentence_embeddings}")
        else:
            sentence_embeddings = mean_pooling(results, tester.input["attention_mask"])

            print("Sentence embeddings:")
            print(sentence_embeddings)

    tester.finalize()
