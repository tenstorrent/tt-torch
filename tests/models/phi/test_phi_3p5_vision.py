# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Phi-3.5-vision: https://huggingface.co/microsoft/Phi-3.5-vision-instruct

import torch
import pytest

from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.phi3.phi_3_5_vision.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        model = self.loader.load_model(dtype_override=torch.bfloat16)
        self.tokenizer = self.loader.tokenizer
        return model

    def _load_inputs(self):
        return self.loader.load_inputs()


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_phi_3p5_vision(record_property, mode, op_by_op):
    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)
    model_name = model_info.name
    model_group = model_info.group.value

    cc = CompilerConfig()
    cc.enable_consteval = True

    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_FE_COMPILATION",
        model_group=model_group,
        reason="Can't lower remainder op in TORCH IR --> TORCH BACKEND IR",
    )

    tester = ThisTester(
        model_name,
        mode,
        loader=loader,
        model_info=model_info,
        compiler_config=cc,
        record_property_handle=record_property,
        is_token_output=True,
        model_group=model_group,
        run_generate=True,
    )

    results = tester.test_model(assert_eval_token_mismatch=False)

    if mode == "eval":
        # Get input length for proper slicing
        test_inputs = loader.load_inputs()
        input_length = test_inputs["input_ids"].shape[1]

        # Use loader's decode_output method
        decoded_output = loader.decode_output(results, input_length=input_length)

        # Get test input from loader for display
        test_input = [
            {"role": "user", "content": "<|image_1|>\nWhat is this image about?"},
        ]

        print(
            f"""
        model_name: {model_name}
        input: {test_input}
        output: {decoded_output}
        """
        )
    tester.finalize()
