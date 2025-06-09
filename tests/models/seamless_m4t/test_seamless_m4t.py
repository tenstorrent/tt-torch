# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/facebook/seamless-m4t-v2-large
import torch
import requests
import pytest
import urllib.request
import io
import scipy
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend

from transformers import AutoProcessor, SeamlessM4TModel, SeamlessM4TConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.modeling_outputs import BaseModelOutput
import torchaudio
import types


class ThisTester(ModelTester):
    def _load_model(self):
        model_name = "facebook/hf-seamless-m4t-large"
        self.config = SeamlessM4TConfig.from_pretrained(model_name, use_cache=False)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.full_model = SeamlessM4TModel.from_pretrained(
            model_name, config=self.config
        )
        # Compile text_decoder submodule only
        return self.full_model.text_decoder

    def _load_inputs(self):
        # Load and format audio input
        url = "https://courses.cs.duke.edu/cps001/spring06/class/06_Sound/sounds/preamble.wav"
        with urllib.request.urlopen(url) as response:
            audio_data = response.read()
        audio_buffer = io.BytesIO(audio_data)
        audio, orig_freq = torchaudio.load(audio_buffer)
        audio = torchaudio.functional.resample(
            audio, orig_freq=orig_freq, new_freq=16_000
        )
        audio_inputs = self.processor(audios=audio, return_tensors="pt")
        # Run encoder to get encoder_hidden_states
        encoder_outputs = self.full_model.speech_encoder(
            input_features=audio_inputs.input_features,
            attention_mask=audio_inputs.attention_mask,
        )
        encoder_hidden_states = encoder_outputs[0]
        # Prepare decoder input IDs
        tokenizer = self.processor.tokenizer
        bos_token_id = tokenizer.bos_token_id
        decoder_input_ids = torch.tensor([[bos_token_id]])
        # Arguments are inputs for the text decoder submodule
        arguments = {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
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
def test_seamless_m4t(record_property, mode, op_by_op):
    model_name = "SeamlessM4T"
    cc = CompilerConfig()
    cc.enable_consteval = True
    # cc.consteval_parameters = True
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
        run_generate=False,
        model_group="red",
    )
    results = tester.test_model()
    if mode == "eval":
        if tester.run_generate:
            sample_rate = tester.model.config.sampling_rate
            # uncomment this to download the output audio
            # scipy.io.wavfile.write(
            #     "out_from_text.wav", rate=sample_rate, data=results[0].numpy().squeeze()
            # )

    tester.finalize()
