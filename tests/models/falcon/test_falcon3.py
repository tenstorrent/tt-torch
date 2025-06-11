# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    OpByOpBackend,
    ModelMetadata,
)


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


FALCON3_VARIANTS = [
    ModelMetadata(model_name="tiiuae/Falcon3-1B-Base"),
    ModelMetadata(model_name="tiiuae/Falcon3-3B-Base", assert_pcc=True),
    ModelMetadata(model_name="tiiuae/Falcon3-7B-Base"),
    ModelMetadata(model_name="tiiuae/Falcon3-10B-Base"),
    ModelMetadata(model_name="tiiuae/Falcon3-1B-Instruct"),
    ModelMetadata(model_name="tiiuae/Falcon3-3B-Instruct"),
    ModelMetadata(model_name="tiiuae/Falcon3-7B-Instruct"),
    ModelMetadata(model_name="tiiuae/Falcon3-10B-Instruct"),
]


@pytest.mark.parametrize("model_info", FALCON3_VARIANTS, ids=lambda x: x.model_name)
@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
@pytest.mark.parametrize(
    "op_by_op_backend",
    [OpByOpBackend.TORCH, OpByOpBackend.STABLEHLO],
    ids=["torch", "stablehlo"],
)
def test_falcon(record_property, model_info, mode, execute_mode, op_by_op_backend):
    if (
        execute_mode == CompileDepth.EXECUTE
        and op_by_op_backend == OpByOpBackend.STABLEHLO
    ):
        pytest.skip("Full graph execution is backend-agnostic")

    model_group = "red"
    cc = CompilerConfig()
    cc.enable_consteval = True
    # consteval_parameters is disabled because it results in a memory related crash

    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
        cc.compile_depth = execute_mode
        cc.op_by_op_backend = op_by_op_backend
    else:
        cc.compile_depth = execute_mode

    if execute_mode == CompileDepth.EXECUTE and model_info.model_name in [
        "tiiuae/Falcon3-7B-Base",
        "tiiuae/Falcon3-10B-Base",
        "tiiuae/Falcon3-7B-Instruct",
        "tiiuae/Falcon3-10B-Instruct",
    ]:
        pytest.skip("Model is too large to fit on single device during execution.")

    tester = ThisTester(
        model_name=model_info.model_name,
        model_info=model_info,
        mode=mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=model_info.assert_pcc,
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
