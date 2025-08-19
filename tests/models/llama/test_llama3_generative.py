# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import time
from tt_torch.tools.utils import CompilerConfig
from tt_torch.tools.verify import calculate_pcc
from tt_torch.tools.device_manager import DeviceManager
from tt_torch.dynamo.backend import backend, BackendOptions
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StaticCache,
)
from tests.utils import clear_dynamo_cache
from transformers.models.llama.modeling_llama import LlamaModel
import os
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import tt_mlir
import numpy as np
import tt_torch.dynamo.sharding_utils as ts


# Control vars

_global_max_cache_len = 64 + 64
tokens_to_generate = 1
hidden_layers = 1
use_static_cache = True


def load_model(model_name="meta-llama/Llama-3.2-3B"):
    # set up the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=True,
    )
    model.config.num_hidden_layers = hidden_layers

    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer


def load_inputs(
    model,
    tokenizer,
    # test_input="This is a sample text from ",
    test_input="I like taking walks in the",
    max_cache_len=_global_max_cache_len,
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

    if use_static_cache:
        args = {
            "input_ids": inputs.input_ids,
            "past_key_values": static_cache,
            "use_cache": True,
            "cache_position": cache_position,
        }
    else:
        args = {"input_ids": inputs.input_ids}
    return args


def setup_xla_environment():
    """Setup XLA environment for tensor parallelism."""
    print("Setting up XLA environment...")
    num_devices = xr.global_runtime_device_count()

    # Basic XLA configuration
    os.environ[
        "ENABLE_AUTO_PARALLEL"
    ] = "TRUE"  # Enables the auto parallel pass in tt-mlir
    os.environ[
        "CONVERT_SHLO_TO_SHARDY"
    ] = "1"  # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ[
        "MESH_SHAPE"
    ] = f"1,{num_devices}"  # Sets the mesh shape used by the auto parallel pass

    # Initialize SPMD
    xr.use_spmd()
    print("XLA environment configured.")


def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.

    Args:
        num_devices: Total number of devices
        mesh_shape: Shape of the device mesh (batch_dim, model_dim)

    Returns:
        Mesh object for SPMD operations
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


@torch.inference_mode()
def test_llama3_generate():
    # Initialize model and inputs
    start_time = time.time()
    model, tokenizer = load_model()
    input_args = load_inputs(model, tokenizer)
    generated_ids = input_args["input_ids"]

    initial_prompt = tokenizer.decode(generated_ids[0].tolist())
    print(f"Initial prompt: '{initial_prompt}'")

    # setup XLA environment and device mesh
    setup_xla_environment()
    mesh = create_device_mesh()

    ts.mark_sharding(input_args["input_ids"], (None, None))
    if use_static_cache:
        ts.mark_sharding(input_args["cache_position"], (None,))

        # apply shardings
        for i, (key, value) in enumerate(
            zip(
                input_args["past_key_values"].key_cache,
                input_args["past_key_values"].value_cache,
            )
        ):
            ts.mark_sharding(key, (None, "model", None, None))
            ts.mark_sharding(value, (None, "model", None, None))

    for layer in model.model.layers:
        ts.mark_sharding(layer.mlp.up_proj.weight, ("model", None))
        ts.mark_sharding(layer.mlp.gate_proj.weight, ("model", None))
        ts.mark_sharding(layer.mlp.down_proj.weight, (None, "model"))

        ts.mark_sharding(layer.self_attn.q_proj.weight, ("model", None))
        ts.mark_sharding(layer.self_attn.k_proj.weight, ("model", None))
        ts.mark_sharding(layer.self_attn.v_proj.weight, ("model", None))
        ts.mark_sharding(layer.self_attn.o_proj.weight, (None, "model"))

    ts.mark_sharding(model.lm_head.weight, ("model", None))

    # Setup compilation
    clear_dynamo_cache()
    cc = CompilerConfig()
    cc.mesh = mesh

    # Consteval disabled due to 4D Causal Attention Mask evaluation getting constant folded in torchfx
    #   due to incorrect tracing of static cache and malformed output missing static cache tensors
    cc.enable_consteval = True
    cc.consteval_parameters = False

    options = BackendOptions()
    options.compiler_config = cc

    # _backend = backend
    _backend = "tt-experimental"

    compiled_model = torch.compile(
        model, backend=_backend, dynamic=False, options=options
    )

    # Token generation with data collection
    generated_tokens = []

    print(initial_prompt, end="", flush=True)

    for i in range(tokens_to_generate):
        # Execute model
        outputs = compiled_model(**input_args)

        # Update inputs for next iteration
        if use_static_cache:
            cache_position = input_args["cache_position"][-1:] + 1

        # Post-processing
        next_token_ids = outputs.logits[:, -1:].argmax(dim=-1)
        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

        # Decode and collect token
        new_token = tokenizer.decode(next_token_ids[0].tolist())
        generated_tokens.append(new_token)
        print(new_token, end="", flush=True)

        if use_static_cache:
            input_args = {
                "input_ids": next_token_ids,
                "past_key_values": input_args["past_key_values"],  # updated in place
                "cache_position": cache_position,
                "use_cache": True,
            }
        else:
            input_args = {"input_ids": next_token_ids}

    # Cleanup
    clear_dynamo_cache()
