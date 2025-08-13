# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.tools.device_manager import DeviceManager
from tt_torch.dynamo.backend import backend, BackendOptions
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StaticCache,
)
from tests.utils import clear_dynamo_cache

_global_max_cache_len = 64 + 64


def load_model(model_name="meta-llama/Llama-3.2-3B"):
    # set up the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=True,
    )
    model.generation_config.cache_implementation = "static"

    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer


def load_inputs(
    tokenizer,
    test_input="I like taking walks in the",
):
    inputs = tokenizer.encode_plus(
        test_input,
        return_tensors="pt",
        truncation=True,
        return_attention_mask=True,
    )

    args = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": 32,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": False,
    }
    return args


@torch.inference_mode()
def main():
    model, tokenizer = load_model()
    input_args = load_inputs(tokenizer)

    clear_dynamo_cache()
    cc = CompilerConfig()
    cc.enable_consteval = False
    cc.consteval_parameters = False

    options = BackendOptions()
    options.compiler_config = cc

    device = DeviceManager.create_parent_mesh_device(mesh_shape=[1, 1])
    options.devices = [device]

    model.forward = torch.compile(
        model.forward, backend="tt", dynamic=False, options=options
    )

    outputs = model.generate(**input_args)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded_output)

    print()  # Add a newline at the end of the output
    DeviceManager.release_parent_device(device)
    clear_dynamo_cache()


if __name__ == "__main__":
    main()
