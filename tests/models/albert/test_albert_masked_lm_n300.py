# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/albert#transformers.AlbertForMaskedLM

from transformers import AutoTokenizer, AlbertForMaskedLM
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        return AlbertForMaskedLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )

    def _load_inputs(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.text = [
            "The capital of [MASK] is Paris."
        ] * 32  # Create a batch of 32 inputs
        self.inputs = self.tokenizer(self.text, return_tensors="pt")
        return self.inputs

    def set_inputs_train(self, inputs):
        return inputs

    def append_fake_loss_function(self, outputs):
        return torch.mean(outputs.logits)

    # TODO: inputs has no grad, how to get it?
    # def get_results_train(self, model, inputs, outputs):
    #     return


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "albert/albert-base-v2",
        "albert/albert-large-v2",
        "albert/albert-xlarge-v2",
        "albert/albert-xxlarge-v2",
    ],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_albert_masked_lm(record_property, model_name, mode, op_by_op):

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

    assert_pcc = (
        True
        if model_name
        in [
            "albert/albert-base-v2",
            "albert/albert-large-v2",
            "albert/albert-xlarge-v2",
        ]
        else False
    )

    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=assert_pcc,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()

    if mode == "eval":
        logits = results.logits
        input_ids = tester.inputs.input_ids
        mask_token_id = tester.tokenizer.mask_token_id
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            # Find the index of [MASK] for this sample
            mask_token_indices = (input_ids[i] == mask_token_id).nonzero(as_tuple=True)[
                0
            ]
            predicted_token_ids = logits[i, mask_token_indices].argmax(axis=-1)
            predicted_tokens = tester.tokenizer.decode(predicted_token_ids)
            print(
                f"Sample {i}: Model: {model_name} | Input: {tester.text[i]} | Mask: {predicted_tokens}"
            )

    tester.finalize()
