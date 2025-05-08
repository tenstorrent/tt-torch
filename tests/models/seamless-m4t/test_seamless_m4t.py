# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# from text
text_inputs = processor(
    text="Hello, my dog is cute", src_lang="eng", return_tensors="pt"
)
audio_array_from_text = (
    model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
)

# from audio
audio, orig_freq = torchaudio.load("tests/models/Dia/preamble.wav")
audio = torchaudio.functional.resample(
    audio, orig_freq=orig_freq, new_freq=16_000
)  # must be a 16 kHz waveform array
audio_inputs = processor(audios=audio, return_tensors="pt")
audio_array_from_audio = (
    model.generate(**audio_inputs, tgt_lang="tur")[0].cpu().numpy().squeeze()
)

import scipy

sample_rate = model.config.sampling_rate
scipy.io.wavfile.write(
    "out_from_text.wav", rate=sample_rate, data=audio_array_from_audio
)
