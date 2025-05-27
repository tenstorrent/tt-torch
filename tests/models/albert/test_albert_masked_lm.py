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
        self.text = "The capital of [MASK] is Paris."
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
        # "albert/albert-xxlarge-v2",
    ],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_albert_masked_lm(
    record_property, model_name, mode, op_by_op, data_parallel_mode
):
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
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
        data_parallel_mode=data_parallel_mode,
    )
    results = tester.test_model()

    if mode == "eval":
        if data_parallel_mode:
            for i in range(len(results)):
                result = results[i]
                # retrieve index of [MASK]
                logits = result.logits
                mask_token_index = (
                    tester.inputs.input_ids == tester.tokenizer.mask_token_id
                )[0].nonzero(as_tuple=True)[0]
                predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
                predicted_tokens = tester.tokenizer.decode(predicted_token_id)

                print(
                    f"Model: {model_name} | Input: {tester.text} | Mask: {predicted_tokens} | (Device {i})"
                )
        else:
            # retrieve index of [MASK]
            logits = results.logits
            mask_token_index = (
                tester.inputs.input_ids == tester.tokenizer.mask_token_id
            )[0].nonzero(as_tuple=True)[0]
            predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
            predicted_tokens = tester.tokenizer.decode(predicted_token_id)

            print(
                f"Model: {model_name} | Input: {tester.text} | Mask: {predicted_tokens}"
            )

    tester.finalize()
