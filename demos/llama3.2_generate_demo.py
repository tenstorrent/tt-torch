# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import backend, BackendOptions
from tt_torch.tools.device_manager import DeviceManager
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StaticCache,
)

from transformers.models.llama.modeling_llama import LlamaModel

import tt_mlir
import time
import argparse
from tests.utils import clear_dynamo_cache

_global_max_cache_len = 64+64


def load_model(model_name="meta-llama/Llama-3.2-3B"):
    # set up the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=True,
    )

    model.config.num_hidden_layers = 28

    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer


def load_inputs(
    model, tokenizer, test_input="This is a sample text from ", max_cache_len=_global_max_cache_len
):
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
    
    print("[James] Manually forwarding attention mask")
    #  Experiment - Generate attention mask using the LlamaModel method
    attention_mask = LlamaModel._prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask=None,  # No initial attention mask
        sequence_length=inputs.input_ids.shape[1],
        target_length=max_cache_len,
        dtype=model.dtype,
        cache_position=cache_position,
        batch_size=batch_size,
    )
    args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "use_cache": True,
        "cache_position": cache_position,
        # "attention_mask": attention_mask
    }
    return args


@torch.inference_mode()
def main():
    model, tokenizer = load_model()
    input_args = load_inputs(model, tokenizer)
    generated_ids = input_args["input_ids"]

    clear_dynamo_cache()
    cc = CompilerConfig()
    cc.enable_consteval = False
    cc.consteval_parameters = False

    options = BackendOptions()
    options.compiler_config = cc
    
    mesh_options = tt_mlir.MeshDeviceOptions()
    device = tt_mlir.open_mesh_device([1, 1], mesh_options)
    options.devices = [device]
    
    buffer_cache = {}
    options.buffer_cache = buffer_cache

    constant_cache = {}
    options.constant_cache = constant_cache

    compare_golden = False  # enable golden comparison - significantly slows down test due to h2d transfer of cache weights
    compiled_model = torch.compile(
        model, backend=backend, dynamic=False, options=options
    )

    tokens_to_generate = 64
    for i in range(tokens_to_generate):
        print("\n===== Decode step", i, "=====\n")
        print(f"Input args to step {i}", input_args)

        start_time = time.time()

        # Execute through backend
        outputs = compiled_model(**input_args)

        next_token_ids = outputs.logits[:, -1:].argmax(dim=-1)

        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
        print(
            "Decoded output so far:",
            "\033[91m",
            tokenizer.decode(generated_ids[0].tolist()),
            "\033[0m",
        )
        print("Time elapsed for this step: ", time.time() - start_time)
        cache_position = input_args["cache_position"][-1:] + 1

        input_args = {
            "input_ids": next_token_ids.to(dtype=torch.int32),
            "past_key_values": input_args["past_key_values"],  # updated in place
            "cache_position": cache_position,
            "use_cache": True,
                # Generate new attention mask for the next token
            # "attention_mask": LlamaModel._prepare_4d_causal_attention_mask_with_cache_position(
            #     attention_mask=None,
            #     sequence_length=1,  # Just one new token
            #     target_length=_global_max_cache_len,
            #     dtype=model.dtype,
            #     cache_position=cache_position,
            #     batch_size=1,
            # ),
        }


if __name__ == "__main__":
    main()
