# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/gpt_neo#overview

from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GenerationConfig
import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        model = GPTNeoForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-125M", torch_dtype=torch.bfloat16
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        return model

    def _load_inputs(self):
        prompt = (
            "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
            "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
            "researchers was the fact that the unicorns spoke perfect English."
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        generation_config = GenerationConfig(
            max_length=100, do_sample=True, temperature=0.9
        )
        arguments = {"input_ids": input_ids, "generation_config": generation_config}
        return arguments

    def set_model_eval(self, model):
        return model


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_gpt_neo(record_property, mode, op_by_op):
    model_name = "GPTNeo"

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
        assert_pcc=False,
        assert_atol=False,
        run_generate=True,  # run model.generate(**inputs)
        is_token_output=True,
    )
    results = tester.test_model(assert_eval_token_mismatch=False)
    if mode == "eval":
        gen_text = tester.tokenizer.batch_decode(results)[0]

    tester.finalize()
