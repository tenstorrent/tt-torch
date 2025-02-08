# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/facebook/musicgen-small

from transformers import AutoProcessor, MusicgenForConditionalGeneration
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth


class ThisTester(ModelTester):
    def _load_model(self):
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small"
        )
        return model.generate

    def _load_inputs(self):
        inputs = self.processor(
            text=[
                "80s pop track with bassy drums and synth",
                "90s rock song with loud guitars and heavy drums",
            ],
            padding=True,
            return_tensors="pt",
        )

        inputs["max_new_tokens"] = 1
        return inputs

    def set_model_eval(self, model):
        return model


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.xfail(
    reason="Fails due to pt2 compile issue when finishing generation, but we can still generate a graph"
)
@pytest.mark.parametrize("op_by_op", [True, False], ids=["op_by_op", "full"])
def test_musicgen_small(record_property, mode, op_by_op):
    model_name = "musicgen_small"
    record_property("model_name", model_name)
    record_property("mode", mode)

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    tester = ThisTester(
        model_name, mode, compiler_config=cc, assert_atol=False, assert_pcc=False
    )
    results = tester.test_model()

    record_property("torch_ttnn", (tester, results))
