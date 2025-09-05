# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/microsoft/speecht5_tts

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts", torch_dtype=torch.bfloat16
        )
        return model.generate_speech

    def _load_inputs(self):
        inputs = self.processor(text="Hello, my dog is cute.", return_tensors="pt")
        # load xvector containing speaker's voice characteristics from a dataset
        speaker_embeddings = torch.zeros((1, 512)).to(torch.bfloat16)
        vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan", torch_dtype=torch.bfloat16
        )
        arguments = {
            "input_ids": inputs["input_ids"],
            "speaker_embeddings": speaker_embeddings,
            "vocoder": vocoder,
        }
        return arguments


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_speecht5_tts(record_property, mode, op_by_op):
    model_name = "speecht5-tts"

    cc = CompilerConfig()
    cc.enable_consteval = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=False,
        assert_pcc=False,
        # is_token_output=True,
    )
    speech = tester.test_model()
    # if mode == "eval":
    #     # Uncomment below if you really want to hear the result.
    #     import soundfile as sf

    #     sf.write("speech.wav", speech.to(torch.float32).numpy(), samplerate=16000)

    tester.finalize()
