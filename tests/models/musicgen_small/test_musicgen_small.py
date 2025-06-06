# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/facebook/musicgen-small

from transformers import AutoProcessor, MusicgenForConditionalGeneration
import pytest
import torch
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small"
        )
        return model

    def _load_inputs(self):
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        inputs = processor(
            text=[
                "80s pop track with bassy drums and synth",
                "90s rock song with loud guitars and heavy drums",
            ],
            padding=True,
            return_tensors="pt",
        )
        pad_token_id = self.framework_model.generation_config.pad_token_id
        decoder_input_ids = (
            torch.ones(
                (
                    inputs.input_ids.shape[0]
                    * self.framework_model.decoder.num_codebooks,
                    1,
                ),
                dtype=torch.long,
            )
            * pad_token_id
        )

        inputs["max_new_tokens"] = 1
        inputs["decoder_input_ids"] = decoder_input_ids
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
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_musicgen_small(record_property, mode, op_by_op, data_parallel_mode):
    model_name = "musicgen_small"
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
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
        assert_atol=False,
        assert_pcc=False,
        record_property_handle=record_property,
        data_parallel_mode=data_parallel_mode,
    )
    results = tester.test_model()
    tester.finalize()
