# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_torch

import os
import sys
import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


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


def setup_xla_environment():
    """Setup XLA environment for tensor parallelism."""
    print("Setting up XLA environment...")

    # Converts the StableHLO emitted by torch-xla to the Shardy dialect.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    # Initialize SPMD
    xr.use_spmd()
    print("XLA environment configured.")


def apply_tensor_parallel_sharding_base(
    base_model: AutoModelForCausalLM, mesh: Mesh
) -> None:
    """
    Apply tensor parallel sharding to the base model - ALL REPLICATED FOR TESTING.
    """
    print("Applying tensor parallel sharding to base model")
    # Apply replication to each transformer layer
    for layer in base_model.layers:
        #         # ========================================
        #         # MLP (Feed-Forward) Layer Sharding - ALL REPLICATED
        #         # ========================================

        #         # gate_up_proj: [32, 2880, 5760] -> replicate all dimensions
        #         xs.mark_sharding(layer.mlp.experts.gate_up_proj, mesh, (None, None, None))

        #         # gate_up_proj_bias: [32, 5760] -> replicate all dimensions
        #         xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, (None, None))

        #         # down_proj: [32, 2880, 2880] -> replicate all dimensions
        #         xs.mark_sharding(layer.mlp.experts.down_proj, mesh, (None, None, None))

        #         # down_proj_bias: [32, 2880] -> replicate all dimensions
        #         xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, (None, None))

        # ========================================
        # Self-Attention Layer Sharding
        # ========================================

        # q_proj: [num_heads * head_dim, hidden_size] -> colwise
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("batch", "model"))

        # k_proj: [num_kv_heads * head_dim, hidden_size] -> colwise
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("batch", "model"))

        # v_proj: [num_kv_heads * head_dim, hidden_size] -> colwise
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("batch", "model"))

        # o_proj: [hidden_size, num_heads * head_dim] -> rowwise
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("model", "batch"))

        # sinks -> local. rowwise
        xs.mark_sharding(layer.self_attn.sinks, mesh, ("model",))


def apply_tensor_parallel_sharding_causal(
    causal_model: AutoModelForCausalLM, mesh: Mesh
) -> None:
    """
    Apply tensor parallel sharding to the causal Llama model (specifically to the MLP,
    self-attention, and LM heads).
    """
    # Move model to XLA device first
    causal_model = causal_model.to(torch_xla.device())

    # Shard the base model first
    apply_tensor_parallel_sharding_base(causal_model.model, mesh)

    # Now shard the LM head colwise
    xs.mark_sharding(causal_model.lm_head.weight, mesh, ("batch", "model"))
    print("Tensor parallel sharding applied successfully!")


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
            value = value.to(torch_xla.device())

            # Replicate inputs to all devices
            xs.mark_sharding(value, mesh, (None, None))

            # Update the dictionary with the sharded tensor
            inputs[key] = value

    return inputs


def run_gpt_oss_tp():
    """
    Run a single forward pass using tensor parallelism.
    """
    print("Setting up xla environment")
    setup_xla_environment()
    mesh = create_device_mesh()

    print("Loading model...")
    gpt_oss = GPT_OSS()
    model = gpt_oss._load_model()
    inputs = gpt_oss._load_inputs()

    print("Sharding inputs...")
    inputs_tp = prepare_inputs(mesh, inputs)

    print("Moving model to torch_xla device")
    apply_tensor_parallel_sharding_causal(model, mesh)

    print("Running Tensor Parallel Inference")
    with torch.no_grad():
        outputs = model(**inputs_tp)
        print("Ran model")
        torch_xla.sync()
        print("torch xla synced")
    outputs.logits = outputs.logits.to("cpu")
    print(outputs)


def main():
    print("Torch-XLA SPMD Tensor Parallelism for GPT-OSS Model")
    print("=" * 50)
    run_gpt_oss_tp()


if __name__ == "__main__":
    sys.exit(main())


# def setup_xla_environment():
#     """Setup XLA environment for tensor parallelism."""
#     print("Setting up XLA environment...")

#     # Converts the StableHLO emitted by torch-xla to the Shardy dialect.
#     os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

#     # Initialize SPMD
#     xr.use_spmd()
#     print("XLA environment configured.")


# def create_device_mesh() -> Mesh:
#     """
#     Create device mesh for tensor parallelism.
#     """
#     num_devices = xr.global_runtime_device_count()
#     mesh_shape = (1, num_devices)
#     device_ids = np.array(range(num_devices))
#     mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
#     print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
#     return mesh


# def _extract_outputs(output_object):
#     """Extract tensors from various output formats"""
#     if isinstance(output_object, torch.Tensor):
#         return (output_object,)
#     elif isinstance(output_object, (int, float)):
#         return (torch.tensor(output_object),)
#     elif isinstance(output_object, (tuple, list)):

#         def flatten_tensor_lists(obj):
#             flattened = []
#             for item in obj:
#                 if isinstance(item, torch.Tensor):
#                     flattened.append(item)
#                 elif isinstance(item, (np.ndarray)):
#                     flattened.append(torch.from_numpy(item))
#                 elif np.isscalar(item):
#                     flattened.append(torch.tensor(item))
#                 elif isinstance(item, (tuple, list)):
#                     flattened.extend(flatten_tensor_lists(item))
#             return flattened

#         flattened_tensors = flatten_tensor_lists(output_object)
#         return tuple(flattened_tensors)
#     elif hasattr(output_object, "to_tuple"):
#         return output_object.to_tuple()

#     raise NotImplementedError(f"Output type {type(output_object)} not supported")


# def verify_outputs(golden, outputs):
#     """
#     Verify outputs against golden reference and print PCC and ATOL

#     Args:
#         golden: Golden reference outputs
#         outputs: Calculated outputs to verify
#     """
#     # Extract tensors from both golden and outputs
#     golden_tensors = _extract_outputs(golden)
#     output_tensors = _extract_outputs(outputs)

#     # Calculate PCC and ATOL for each tensor pair
#     pccs = []
#     atols = []

#     for i, (golden_tensor, output_tensor) in enumerate(
#         zip(golden_tensors, output_tensors)
#     ):
#         pcc = calculate_pcc(golden_tensor, output_tensor)
#         atol = calculate_atol(golden_tensor, output_tensor)

#         pccs.append(pcc)
#         atols.append(atol)

#         print(f"Output {i}: PCC={pcc:.6f}, ATOL={atol:.6e}")

#     # Print summary
#     if pccs:
#         print(f"Summary: Min PCC={min(pccs):.6f}, Max ATOL={max(atols):.6e}")

#     return pccs, atols


# def move_input_to_device(input, device: Literal["cpu", "xla"]):
#     assert device == "cpu" or device == "xla", f"Invalid device: {device}"
#     input["input_ids"] = input["input_ids"].to(device)
#     input["attention_mask"] = input["attention_mask"].to(device)
#     return


# def move_output_to_device(output, device: Literal["cpu", "xla"]):
#     assert device == "cpu" or device == "xla", f"Invalid device: {device}"
#     output.logits = output.logits.to(device)
#     return


# def run_golden(model, input):
#     device = "cpu"
#     model.to(device)
#     move_input_to_device(input, device)
#     torch._dynamo.reset()
#     with torch.no_grad():
#         return model(**input)


# def apply_tensor_parallel_sharding_base(
#     base_model: AutoModelForCausalLM, mesh: Mesh
# ) -> None:
#     """
#     Apply tensor parallel sharding to the base model - ALL REPLICATED FOR TESTING.
#     """
#     # Apply replication to each transformer layer
#     for layer in base_model.layers:
#         # ========================================
#         # MLP (Feed-Forward) Layer Sharding - ALL REPLICATED
#         # ========================================

#         # gate_up_proj: [32, 2880, 5760] -> replicate all dimensions
#         xs.mark_sharding(layer.mlp.experts.gate_up_proj, mesh, (None, None, None))

#         # gate_up_proj_bias: [32, 5760] -> replicate all dimensions
#         xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, (None, None))

#         # down_proj: [32, 2880, 2880] -> replicate all dimensions
#         xs.mark_sharding(layer.mlp.experts.down_proj, mesh, (None, None, None))

#         # down_proj_bias: [32, 2880] -> replicate all dimensions
#         xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, (None, None))

#         # ========================================
#         # Self-Attention Layer Sharding - ALL REPLICATED
#         # ========================================

#         # q_proj: [num_heads * head_dim, hidden_size] -> replicate all dimensions
#         xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, (None, None))

#         # k_proj: [num_kv_heads * head_dim, hidden_size] -> replicate all dimensions
#         xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, (None, None))

#         # v_proj: [num_kv_heads * head_dim, hidden_size] -> replicate all dimensions
#         xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, (None, None))

#         # o_proj: [hidden_size, num_heads * head_dim] -> replicate all dimensions
#         xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, None))

# def apply_tensor_parallel_sharding_causal(model, mesh: Mesh):
#     model = model.to(xla_device)
#     # lm_head: replicate all dimensions
#     xs.mark_sharding(model.lm_head.weight, mesh, (None, None))
#     apply_tensor_parallel_sharding_base(model.model, mesh)
#     print("Tensor parallel sharding applied successfully! (ALL REPLICATED)")


# def prepare_inputs(mesh: Mesh, inputs: dict) -> dict:
#     """
#     Prepare input dictionary by moving tensors to the XLA device and
#     replicating them across the device mesh.
#     """
#     print("Preparing inputs for TP:")

#     # Move all tensors in the dictionary to the XLA device and apply sharding
#     for key, value in inputs.items():
#         if isinstance(value, torch.Tensor):
#             print(
#                 f"  - Sharding tensor '{key}': "
#                 f"batch_size={value.shape[0]}, seq_length={value.shape[1]}"
#             )

#             # Move to XLA device
#             value = value.to(xla_device)

#             # Replicate inputs to all devices
#             xs.mark_sharding(value, mesh, (None, None))

#             # Update the dictionary with the sharded tensor
#             inputs[key] = value

#     return inputs


# class GPT_OSS:
#     def __init__(self):
#         self.model_name = "openai/gpt-oss-20b"

#     def _load_model(self):
#         config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
#         config.quantization_config["quant_method"] = "none"
#         config.num_hidden_layers = 1
#         config.use_cache = False
#         model = AutoModelForCausalLM.from_pretrained(
#             self.model_name,
#             config=config,
#             torch_dtype=torch.bfloat16,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,
#             attn_implementation="eager",
#         )
#         model.eval()
#         return model

#     def _load_inputs(self):
#         tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         messages = [
#             {"role": "user", "content": "Who are you?"},
#         ]
#         inputs = tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             tokenize=True,
#             return_dict=True,
#             return_tensors="pt",
#         )
#         return inputs


# def main():

#     setup_xla_environment()
#     mesh = create_device_mesh()

#     test_basic_xla_sync()
#     breakpoint()

#     gpt_oss = GPT_OSS()
#     model = gpt_oss._load_model()
#     inputs = gpt_oss._load_inputs()

#     # Apply sharding to model (all replicated)
#     apply_tensor_parallel_sharding_causal(model, mesh)

#     # Prepare inputs for tensor parallel execution
#     inputs_tp = prepare_inputs(mesh, inputs)

#     torch._dynamo.reset()
#     with torch.no_grad():
#         outputs_tp = model(**inputs_tp)
#         print("Ran model")
#         torch_xla.sync()
#         print("Synchronized outputs")

#     move_output_to_device(outputs_tp, "cpu")
#     print("Test completed successfully!")


# if __name__ == "__main__":
#     main()
