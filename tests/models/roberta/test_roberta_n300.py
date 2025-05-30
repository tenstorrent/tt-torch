# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/en/model_doc/xlm-roberta#transformers.XLMRobertaForMaskedLM

from transformers import AutoTokenizer, XLMRobertaForMaskedLM
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
        model = XLMRobertaForMaskedLM.from_pretrained(
            "FacebookAI/xlm-roberta-base", torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):
        inputs = [
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
            "The capital of France is <mask>.",
        ]
        inputs = self.tokenizer(inputs, return_tensors="pt")
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
def test_roberta(record_property, mode, op_by_op):
    model_name = "RoBERTa"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.automatic_parallelization = True
    cc.mesh_shape = [1, 2]
    cc.dump_debug = True
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
        assert_pcc=True,
        assert_atol=False,
    )
    results = tester.test_model()
    if mode == "eval":
        logits = results.logits
        input_ids = tester.inputs.input_ids
        mask_token_id = tester.tokenizer.mask_token_id
        batch_size = input_ids.shape[0]
        outputs = []
        for i in range(batch_size):
            mask_token_indices = (input_ids[i] == mask_token_id).nonzero(as_tuple=True)[
                0
            ]
            predicted_token_ids = logits[i, mask_token_indices].argmax(axis=-1)
            output = tester.tokenizer.decode(predicted_token_ids)
            outputs.append(output)

    tester.finalize()
