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
        self.tokenizer = WhisperProcessor.from_pretrained(
            "openai/whisper-small", torch_dtype=torch.bfloat16
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small", torch_dtype=torch.bfloat16
        )
        model.config.forced_decoder_ids = None
        model.config.use_cache = False
        return model

    def _load_inputs(self):
        # load dummy dataset and read audio files
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        sample = ds[0]["audio"]
        input_features = self.tokenizer(
            sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
        ).to(torch.bfloat16)
        input_features["decoder_input_ids"] = torch.tensor([[50258]])
        return input_features


@pytest.mark.parametrize(
    "mode",
    ["eval"],
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

    # TODO Enable checking - https://github.com/tenstorrent/tt-torch/issues/593
    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=True,
        assert_atol=False,
    )
    tester.test_model()
    tester.finalize()
