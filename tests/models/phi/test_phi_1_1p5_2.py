# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Phi 1: https://huggingface.co/microsoft/phi-1
# Phi 1.5: https://huggingface.co/microsoft/phi-1_5
# Phi 2: https://huggingface.co/microsoft/phi-2

import torch
import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        return model

    def _load_inputs(self):
        input_str = '''def print_prime(n):
                        """
                        Print all primes between 1 and n
                        """'''
        self.test_input = input_str
        inputs = self.tokenizer(
            input_str, return_tensors="pt", return_attention_mask=False
        )
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name", ["microsoft/phi-1", "microsoft/phi-1.5", "microsoft/phi-2"]
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_phi(record_property, model_name, mode, op_by_op, data_parallel_mode):
    model_group = "red"
    cc = CompilerConfig()
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        is_token_output=True,
        model_group=model_group,
        run_generate=True,  # run model.generate(**inputs)
        data_parallel_mode=data_parallel_mode,
    )

    # TODO - Enable checking - https://github.com/tenstorrent/tt-torch/issues/528
    results = tester.test_model(assert_eval_token_mismatch=False)

    def print_result(result):
        decoded_output = tester.tokenizer.decode(result[0])
        print(
            f"""
        model_name: {model_name}
        input: {tester.test_input}
        output: {decoded_output}
        """
        )

    if mode == "eval":
        ModelTester.print_outputs(results, data_parallel_mode, print_result)

    tester.finalize()
