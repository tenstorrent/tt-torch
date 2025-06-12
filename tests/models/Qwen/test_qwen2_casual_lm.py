# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/Qwen/Qwen2.5-1.5B

from transformers import AutoTokenizer, Qwen2ForCausalLM, GenerationConfig
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
import tt_mlir


class ThisTester(ModelTester):
    def _load_model(self):
        model = Qwen2ForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):

        self.text = "Hey, are you conscious? Can you talk to me?"
        input_ids = self.tokenizer(self.text, return_tensors="pt").input_ids
        generation_config = GenerationConfig(max_length=30)
        arguments = {"input_ids": input_ids, "generation_config": generation_config}
        return arguments


@pytest.mark.parametrize(
    "mode",
    ["eval", "train"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen2.5-1.5B",
    ],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_qwen2_casual_lm(record_property, model_name, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    # TODO: Remove this once PCC ATOL is fixed on blackhole runners - https://github.com/tenstorrent/tt-torch/issues/1003
    assert_pcc = tt_mlir.get_arch() != tt_mlir.Arch.BLACKHOLE

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=assert_pcc,
        assert_atol=False,
        run_generate=False,
        required_pcc=0.86,
    )

    results = tester.test_model()

    if mode == "eval":
        logits = results.logits if hasattr(results, "logits") else results[0]
        token_ids = torch.argmax(logits, dim=-1)
        gen_text = tester.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(f"Model: {model_name} | Input: {tester.text} | Decoded Text: {gen_text}")

    tester.finalize()
