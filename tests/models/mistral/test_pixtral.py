# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


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


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_pixtral(record_property, mode, op_by_op):
    pytest.skip()  # https://github.com/tenstorrent/tt-torch/issues/864
    model_name = "mistral-community/pixtral-12b"
    model_group = "red"
    cc = CompilerConfig()
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
    )

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
        assert_pcc=False,
        assert_atol=False,
    )
    results = tester.test_model()
    tester.finalize()
