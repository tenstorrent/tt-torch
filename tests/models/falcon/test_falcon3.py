# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", torch_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):
        self.prompt = "Hey, are you conscious? Can you talk to me?"
        inputs = self.tokenizer(
            self.prompt, return_tensors="pt", return_token_type_ids=False
        )
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "tiiuae/Falcon3-1B-Base",
        "tiiuae/Falcon3-3B-Base",
        "tiiuae/Falcon3-7B-Base",
        "tiiuae/Falcon3-10B-Base",
        "tiiuae/Falcon3-1B-Instruct",
        "tiiuae/Falcon3-3B-Instruct",
        "tiiuae/Falcon3-7B-Instruct",
        "tiiuae/Falcon3-10B-Instruct",
    ],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_falcon(record_property, model_name, mode, op_by_op):
    model_group = "red"
    cc = CompilerConfig()
    cc.enable_consteval = True
    # consteval_parameters is disabled because it results in a memory related crash
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_RUNTIME",
        reason="Model is too large to fit on single device during execution.",
        model_group=model_group,
        model_name_filter=[
            "tiiuae/Falcon3-7B-Base",
            "tiiuae/Falcon3-10B-Base",
            "tiiuae/Falcon3-7B-Instruct",
            "tiiuae/Falcon3-10B-Instruct",
        ],
    )

    assert_pcc = (
        True
        if model_name
        in [
            "tiiuae/Falcon3-3B-Base",
            "tiiuae/Falcon3-1B-Base",
        ]
        else False
    )

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=assert_pcc,
        assert_atol=False,
        model_group=model_group,
        run_generate=True,  # run model.generate(**inputs)
    )
    results = tester.test_model()

    if mode == "eval":
        output = tester.tokenizer.batch_decode(
            results, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    tester.finalize()
