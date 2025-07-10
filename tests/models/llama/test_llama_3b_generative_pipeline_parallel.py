# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig
from tt_torch.tools.device_manager import DeviceManager
from tt_torch.dynamo.backend import backend, BackendOptions
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StaticCache,
    LlamaConfig,
)
from tests.utils import clear_dynamo_cache
from accelerate import infer_auto_device_map


def load_model(model_name):
    # Create a custom config with only 2 layers (one for each device)
    # config = LlamaConfig.from_pretrained(model_name)
    # config.num_hidden_layers = 2

    # Load model with custom config (remove use_cache from here)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # config=config,
        # ignore_mismatched_sizes=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer


def load_inputs(model, tokenizer):
    test_input = "This is a sample text from "
    max_cache_len = 64 + 64
    batch_size = 1
    inputs = tokenizer.encode_plus(
        test_input,
        return_tensors="pt",
        truncation=True,
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


@torch.inference_mode()
def test_llama_3b_generative_pipeline_parallel(record_property):
    clear_dynamo_cache()
    cc = CompilerConfig()
    cc.enable_consteval = False
    cc.consteval_parameters = False

    model_name = "meta-llama/Llama-3.2-3B"

    model, tokenizer = load_model(model_name)
    input_args = load_inputs(model, tokenizer)
    generated_ids = input_args["input_ids"]
    print(tokenizer.decode(generated_ids[0].tolist()), end="", flush=True)

    parent_device = DeviceManager.create_parent_mesh_device([1, 2])

    # Create submeshes that target different devices
    device1 = DeviceManager.create_sub_mesh_device(parent_device, (0, 0))
    device2 = DeviceManager.create_sub_mesh_device(parent_device, (0, 1))

    # Create a custom device map to test split of kv cache attributes across devices
    # Uncomment the lines at the start of load_model to test this device map.
    # device_map = {
    #     "model.embed_tokens": 0,
    #     "model.rotary_emb": 0,
    #     "model.layers.0": 0,  # one layer on device 0
    #     "model.layers.1": 1,  # one layer on device 1
    #     "model.norm": 1,
    #     "lm_head": 1,
    # }

    dont_split = (
        model._no_split_modules if hasattr(model, "_no_split_modules") else None
    )
    # Use 4GiB to split the graph evenly for this model. The model can fit on one 12 GiB device, but we want to test
    # the pipeline parallelism.
    device_map = infer_auto_device_map(
        model, max_memory={0: "4GiB", 1: "4GiB"}, no_split_module_classes=dont_split
    )

    # This device map assigns the last module to the first device. Since sort_device_map doesn't handle the case where
    # input_device and input_key are None, it won't catch the device_map not being in topological order and there will
    # be an issue with the first device receiving an INTER_DEVICE input.
    # TODO: Fix sort_device_map to handle input_device = None

    # Once the above issue is fixed, the entire model will run on the first device which is not good.
    # TODO: Adjust sort_device_map to ensure that some modules remain on later devices.

    options = BackendOptions()
    options.compiler_config = cc
    cc.device_map = device_map
    options.devices = [device1, device2]

    buffer_cache = {}
    options.buffer_cache = buffer_cache

    constant_cache = {}
    options.constant_cache = constant_cache

    compiled_model = torch.compile(model, backend="tt", dynamic=False, options=options)

    # up to _global_max_cache_len - input_args["input_ids"].shape[1]
    tokens_to_generate = 32

    for i in range(tokens_to_generate):
        outputs = compiled_model(**input_args)
        next_token_ids = outputs.logits[:, -1:].argmax(dim=-1)
        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
        print(tokenizer.decode(next_token_ids[0].tolist()), end="", flush=True)

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
