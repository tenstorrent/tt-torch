# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.tools.device_manager import DeviceManager
from tt_torch.dynamo.backend import BackendOptions
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StaticCache,
)
from tests.utils import clear_dynamo_cache
from accelerate import infer_auto_device_map
import pytest


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", torch_dtype=torch.bfloat16
    )
    tokenizer.pad_token = tokenizer.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    for param in m.parameters():
        param.requires_grad = False
    return m.eval(), tokenizer


def load_inputs(model, tokenizer):
    test_input = "I enjoy walking in the"
    max_cache_len = 64 + 64
    batch_size = 1
    inputs = tokenizer.encode_plus(
        test_input,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True,
    )

    # set up static cache
    static_cache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=model.device,
        dtype=model.dtype,
    )

    cache_position = torch.arange(0, inputs.input_ids.shape[1])

    args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "use_cache": True,
        "cache_position": cache_position,
    }
    return args


@pytest.mark.skip(
    reason="Test hangs during execution - GitHub issue: https://github.com/tenstorrent/tt-torch/issues/1238"
)
@torch.inference_mode()
def test_llama_7b_generative_pipeline_parallel():
    clear_dynamo_cache()
    cc = CompilerConfig()
    cc.enable_consteval = False
    cc.consteval_parameters = False

    model_name = "huggyllama/llama-7b"

    model, tokenizer = load_model(model_name)
    input_args = load_inputs(model, tokenizer)
    generated_ids = input_args["input_ids"]
    print(tokenizer.decode(generated_ids[0].tolist()), end="", flush=True)

    parent_device = DeviceManager.create_parent_mesh_device([1, 2])

    # Create submeshes that target different devices
    device1 = DeviceManager.create_sub_mesh_device(parent_device, (0, 0))
    device2 = DeviceManager.create_sub_mesh_device(parent_device, (0, 1))

    dont_split = (
        model._no_split_modules if hasattr(model, "_no_split_modules") else None
    )
    # The devices have 12GB of memory each, but we set it to 11GB to ensure
    # there's enough room for activation tensors and other overhead.
    device_map = infer_auto_device_map(
        model, max_memory={0: "11GiB", 1: "11GiB"}, no_split_module_classes=dont_split
    )

    options = BackendOptions()
    options.compiler_config = cc
    cc.device_map = device_map
    options.devices = [device1, device2]

    buffer_cache = {}
    options.buffer_cache = buffer_cache

    constant_cache = {}
    options.constant_cache = constant_cache

    compiled_model = torch.compile(
        model, backend="tt-legacy", dynamic=False, options=options
    )

    # up to _global_max_cache_len - input_args["input_ids"].shape[1]
    tokens_to_generate = 32

    for i in range(tokens_to_generate):
        outputs = compiled_model(**input_args)
        next_token_ids = outputs.logits[:, -1:].argmax(dim=-1)
        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
        print(tokenizer.decode(next_token_ids[0].tolist()), end=" ", flush=True)

        cache_position = input_args["cache_position"][-1:] + 1
        input_args = {
            "input_ids": next_token_ids.to(dtype=torch.int32),
            "past_key_values": input_args["past_key_values"],  # updated in place
            "cache_position": cache_position,
            "use_cache": True,
        }
    print()  # Add a newline at the end of the output
    DeviceManager.release_sub_mesh_device(device1)
    DeviceManager.release_sub_mesh_device(device2)
    DeviceManager.release_parent_device(parent_device)
    clear_dynamo_cache()
