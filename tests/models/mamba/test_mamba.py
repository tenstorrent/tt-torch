# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/state-spaces/mamba-2.8b-hf

from transformers import MambaForCausalLM, AutoTokenizer, GenerationConfig
import pytest
from tests.utils import ModelTester
import torch
import types
from transformers.models.mamba.modeling_mamba import MambaCache


def new_cache_init(self, config, max_batch_size, max_length, device, dtype):
    self.max_batch_size = max_batch_size
    self.max_length = max_length
    self.dtype = dtype
    self.device = device

    batch_shape = (config.num_hidden_layers, max_batch_size)

    conv_states = torch.zeros(
        *batch_shape,
        config.hidden_size,
        config.conv_kernel - 1,
        device=device,
        dtype=dtype,
    )
    self.register_buffer("conv_states", conv_states, persistent=False)

    ssm_state_shape = (
        config.num_hidden_layers,
        max_batch_size,
        config.intermediate_size,
        config.state_size,
    )
    ssm_states = torch.zeros(ssm_state_shape, device=device, dtype=dtype)
    self.register_buffer("ssm_states", ssm_states, persistent=False)


# Replace the cache initialization
MambaCache.__init__ = new_cache_init


class ThisTester(ModelTester):
    def _load_model(self):
        model = MambaForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )

        # Correctly override generate method
        original_generate = model.generate

        def generate_without_cache(self, **kwargs):
            kwargs["use_cache"] = False
            return original_generate(**kwargs)

        model.generate = types.MethodType(generate_without_cache, model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        return model.generate

    def _load_inputs(self):
        prompt = "Hey how are you doing?"
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        generation_config = GenerationConfig(max_new_tokens=10, use_cache=False)
        arguments = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "use_cache": False,
        }
        return arguments

    def set_model_eval(self, model):
        return model


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "state-spaces/mamba-790m-hf",
        "state-spaces/mamba-2.8b-hf",
        "state-spaces/mamba-1.4b-hf",
        "state-spaces/mamba-370m-hf",
    ],
)
def test_mamba(record_property, mode, model_name):
    record_property("model_name", model_name)
    record_property("mode", mode)
    tester = ThisTester(model_name, mode)
    results = tester.test_model()
    if mode == "eval":
        gen_text = tester.tokenizer.batch_decode(results)
        print("Generated text: ", gen_text)

    record_property("torch_ttnn", (tester, results))
