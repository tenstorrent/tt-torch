# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, ModelMetadata


class ThisTester(ModelTester):
    def _load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):
        # Set up sample input
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "content": "Can this animal"},
                    {"type": "image", "url": "https://picsum.photos/id/237/200/300"},
                    {"type": "text", "content": "live here?"},
                    {
                        "type": "image",
                        "url": "https://picsum.photos/seed/picsum/200/300",
                    },
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.framework_model.device)
        return inputs


PIXTRAL_VARIANTS = [
    ModelMetadata(model_name="mistral-community/pixtral-12b", model_group="red")
]


@pytest.mark.parametrize("model_info", PIXTRAL_VARIANTS, ids=lambda x: x.model_name)
@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
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
