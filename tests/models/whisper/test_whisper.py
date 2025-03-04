# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        # load model and processor
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-small", torch_dtype=torch.bfloat16
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small", torch_dtype=torch.bfloat16
        )
        model.config.forced_decoder_ids = None
        return model.generate

    def _load_inputs(self):
        # load dummy dataset and read audio files
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        sample = ds[0]["audio"]
        input_features = self.processor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
        ).input_features
        return input_features.to(torch.bfloat16)

    def run_model(self, model, input_features):
        # generate token ids
        predicted_ids = model(input_features)
        # decode token ids to text
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return transcription

    def set_model_eval(self, model):
        return model


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.xfail(
    reason="Fails due to pt2 compile issue when finishing generation, but we can still generate a graph"
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_whisper(record_property, mode, op_by_op):
    model_name = "Whisper"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        is_token_output=True,
    )
    tester.test_model()
    tester.finalize()
