# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    OpByOpBackend,
    calculate_pcc,
    calculate_atol,
)
from tt_torch.dynamo.backend import BackendOptions


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


def main():
    config = AutoConfig.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    config.quantization_config["quant_method"] = "none"
    config.num_hidden_layers = 1
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
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
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    options = BackendOptions()
    options.compiler_config = cc

    with torch.no_grad():
        torch._dynamo.reset()
        tt_model = torch.compile(
            model, backend="tt-experimental", dynamic=False, options=options
        )
        calculated = tt_model(**inputs)
        torch._dynamo.reset()
        golden = model(**inputs)
        verify_outputs(golden, calculated)


if __name__ == "__main__":
    main()
