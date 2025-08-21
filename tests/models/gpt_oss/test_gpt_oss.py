# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np
import os
from typing import Literal

import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from tt_torch.tools.utils import (
    OpByOpBackend,
    calculate_pcc,
    calculate_atol,
)

xla_device = torch_xla.device()


def setup_xla_environment():
    """Setup XLA environment for tensor parallelism."""
    print("Setting up XLA environment...")

    # Converts the StableHLO emitted by torch-xla to the Shardy dialect.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    # Initialize SPMD
    xr.use_spmd()
    print("XLA environment configured.")


def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


def _extract_outputs(output_object):
    """Extract tensors from various output formats"""
    if isinstance(output_object, torch.Tensor):
        return (output_object,)
    elif isinstance(output_object, (int, float)):
        return (torch.tensor(output_object),)
    elif isinstance(output_object, (tuple, list)):

        def flatten_tensor_lists(obj):
            flattened = []
            for item in obj:
                if isinstance(item, torch.Tensor):
                    flattened.append(item)
                elif isinstance(item, (np.ndarray)):
                    flattened.append(torch.from_numpy(item))
                elif np.isscalar(item):
                    flattened.append(torch.tensor(item))
                elif isinstance(item, (tuple, list)):
                    flattened.extend(flatten_tensor_lists(item))
            return flattened

        flattened_tensors = flatten_tensor_lists(output_object)
        return tuple(flattened_tensors)
    elif hasattr(output_object, "to_tuple"):
        return output_object.to_tuple()

    raise NotImplementedError(f"Output type {type(output_object)} not supported")


def verify_outputs(golden, outputs):
    """
    Verify outputs against golden reference and print PCC and ATOL

    Args:
        golden: Golden reference outputs
        outputs: Calculated outputs to verify
    """
    # Extract tensors from both golden and outputs
    golden_tensors = _extract_outputs(golden)
    output_tensors = _extract_outputs(outputs)

    # Calculate PCC and ATOL for each tensor pair
    pccs = []
    atols = []

    for i, (golden_tensor, output_tensor) in enumerate(
        zip(golden_tensors, output_tensors)
    ):
        pcc = calculate_pcc(golden_tensor, output_tensor)
        atol = calculate_atol(golden_tensor, output_tensor)

        pccs.append(pcc)
        atols.append(atol)

        print(f"Output {i}: PCC={pcc:.6f}, ATOL={atol:.6e}")

    # Print summary
    if pccs:
        print(f"Summary: Min PCC={min(pccs):.6f}, Max ATOL={max(atols):.6e}")

    return pccs, atols


def move_input_to_device(input, device: Literal["cpu", "xla"]):
    assert device == "cpu" or device == "xla", f"Invalid device: {device}"
    input["input_ids"] = input["input_ids"].to(device)
    input["attention_mask"] = input["attention_mask"].to(device)
    return


def move_output_to_device(output, device: Literal["cpu", "xla"]):
    assert device == "cpu" or device == "xla", f"Invalid device: {device}"
    output.logits = output.logits.to(device)
    return


def run_golden(model, input):
    device = "cpu"
    model.to(device)
    move_input_to_device(input, device)
    torch._dynamo.reset()
    with torch.no_grad():
        return model(**input)


def apply_tensor_parallel_sharding_base(
    base_model: AutoModelForCausalLM, mesh: Mesh
) -> None:
    """
    Apply tensor parallel sharding to the GPT-OSS base model.
    """
    # Apply sharding to each transformer layer
    for layer in base_model.layers:
        # ========================================
        # Self-Attention Layer Sharding
        # ========================================

        # q_proj: [num_heads * head_dim, hidden_size] -> shard dim 0
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", "batch"))

        # k_proj: [num_kv_heads * head_dim, hidden_size] -> shard dim 0
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", "batch"))

        # v_proj: [num_kv_heads * head_dim, hidden_size] -> shard dim 0
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", "batch"))

        # o_proj: [hidden_size, num_heads * head_dim] -> shard dim 1
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))

        # ========================================
        # MLP (Feed-Forward) Layer Sharding
        # ========================================
        # These layers are handled by `grouped_gemm` and `ep_router`
        # as indicated by the plan, so they do not require explicit
        # xs.mark_sharding calls on the weights themselves.
        # The sharding logic for these layers is implicitly managed by the
        # Shardy runtime based on the router and experts configuration.
        #
        # For 'ep_router' and 'grouped_gemm', the sharding is typically
        # applied at the operation level within the backend.

    print("Base model sharding applied successfully.")


def apply_tensor_parallel_sharding_causal(model, mesh: Mesh):
    model = model.to(xla_device)
    xs.mark_sharding(model.lm_head.weight, mesh, ("model", "batch"))
    apply_tensor_parallel_sharding_base(model.model, mesh)
    print("Tensor parallel sharding applied successfully!")


def prepare_inputs(mesh: Mesh, inputs: dict) -> dict:
    """
    Prepare input dictionary by moving tensors to the XLA device and
    replicating them across the device mesh.
    """
    print("Preparing inputs for TP:")

    # Move all tensors in the dictionary to the XLA device and apply sharding
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(
                f"  - Sharding tensor '{key}': "
                f"batch_size={value.shape[0]}, seq_length={value.shape[1]}"
            )

            # Move to XLA device
            value = value.to(xla_device)

            # Replicate inputs to all devices
            xs.mark_sharding(value, mesh, (None, None))

            # Update the dictionary with the sharded tensor
            inputs[key] = value

    return inputs


class GPT_OSS:
    def __init__(self):
        self.model_name = "openai/gpt-oss-20b"

    def _load_model(self):
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        config.quantization_config["quant_method"] = "none"
        config.num_hidden_layers = 1
        config.use_cache = False
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model.eval()
        return model

    def _load_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        messages = [
            {"role": "user", "content": "Who are you?"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs


def main():

    setup_xla_environment()
    mesh = create_device_mesh()

    gpt_oss = GPT_OSS()
    model = gpt_oss._load_model()
    inputs = gpt_oss._load_inputs()

    # Apply sharding to model
    apply_tensor_parallel_sharding_causal(model, mesh)

    # Prepare inputs for tensor parallel execution
    inputs_tp = prepare_inputs(mesh, inputs)

    torch._dynamo.reset()
    with torch.no_grad():
        outputs_tp = model(**inputs_tp)
        torch_xla.sync()

    move_output_to_device(outputs_tp, "cpu")
    # # Run on cpu
    # golden = run_golden(model, input)

    # # Run on xla
    # torch._dynamo.reset()
    # model = model.to(xla_device)
    # move_input_to_device(input, xla_device)
    # with torch.no_grad():
    #     output = model(**input)

    # # PCC/ Atol check
    # device = "cpu"
    # move_output_to_device(output, device)
    # verify_outputs(golden, output)


if __name__ == "__main__":
    main()
