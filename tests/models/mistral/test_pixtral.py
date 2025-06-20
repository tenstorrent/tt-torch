# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from transformers import LlavaForConditionalGeneration  # , AutoProcessor
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, ModelMetadata


class ThisTester(ModelTester):
    def _load_model(self):
        # self.processor = AutoProcessor.from_pretrained(self.model_name)
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):
        # https://github.com/tenstorrent/tt-torch/issues/904
        inputs = {
            "input_ids": torch.tensor(
                [[1, 3, 12483, 1593, 11386, 10, 51883, 3226, 1063, 10, 4]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long
            ),
        }
        return inputs


PIXTRAL_VARIANTS = [
    ModelMetadata(
        model_name="mistral-community/pixtral-12b",
        model_group="red",
    )
]


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize("model_info", PIXTRAL_VARIANTS, ids=lambda x: x.model_name)
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
def test_pixtral(record_property, model_info, mode, execute_mode):
    pytest.skip()  # https://github.com/tenstorrent/tt-torch/issues/864
    cc = CompilerConfig()
    cc.op_by_op_backend = model_info.op_by_op_backend
    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
        cc.compile_depth = execute_mode
    else:
        cc.compile_depth = model_info.compile_depth

    skip_full_eval_test(
        record_property,
        cc,
        model_info.model_name,
        bringup_status="FAILED_RUNTIME",
        reason="Model is too large to fit on single device during execution.",
        model_group=model_info.model_group,
    )

    tester = ThisTester(
        model_name=model_info.model_name,
        model_info=model_info,
        mode=mode,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_info.model_group,
        assert_pcc=model_info.assert_pcc,
        assert_atol=model_info.assert_atol,
    )
    results = tester.test_model()
    tester.finalize()
