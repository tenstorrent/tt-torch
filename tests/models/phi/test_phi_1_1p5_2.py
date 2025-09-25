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
def test_phi(record_property, model_name, mode, op_by_op):
    model_group = "red"
    cc = CompilerConfig()
    cc.enable_consteval = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    # FIXME 44GB memory usage w/ experimental backend and consteval in phi2
    # fails with tt-experimental - https://github.com/tenstorrent/tt-torch/issues/1108
    backend = "tt-legacy" if model_name == "microsoft/phi-2" else "tt-experimental"

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
        required_pcc=0.85
        if model_name == "microsoft/phi-1"
        else 0.92,  # PCC drop observed around Jul 17, follow up in https://github.com/tenstorrent/tt-torch/issues/1070
        run_generate=False,
        assert_atol=False,
        backend=backend,
    )

    results = tester.test_model()

    if mode == "eval":
        logits = results.logits if hasattr(results, "logits") else results[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        input_ids = tester._load_inputs()["input_ids"]
        output_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
        decoded_output = tester.tokenizer.decode(output_ids[0])
        print(
            f"""
        model_name: {model_name}
        input: {tester.test_input}
        output: {decoded_output}
        """
        )
    tester.finalize()
