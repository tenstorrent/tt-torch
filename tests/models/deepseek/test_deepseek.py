# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import os
import json
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.dynamic_module_utils import get_imports


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-V3"
### Model specs
# num_hidden_layers (currently 61)
# num_attention_heads (currently 128)
# hidden_size (not shown in your config but is likely 8192 based on the model)
###
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load only specific layers
    # Create a custom loading function
    model_name = "deepseek-ai/DeepSeek-V3"  # This might be the generic model name
    config_path = "tests/models/deepseek/config_16B.json"

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config_dict,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        # Load only specific layers
        max_memory={"cpu": "2GB"},  # Limit memory usage
        low_cpu_mem_usage=True,
        offload_folder="offload",  # Temporary directory for offloading
        offload_state_dict=True,  # Enable state dict offloading
    )

    # Reduce model size after loading
    keep_layers = [0, 1, 30, 60]  # Keep first two, middle, and last layer
    new_state_dict = OrderedDict()

    for key, value in model.state_dict().items():
        if "layers." in key:
            layer_num = int(key.split("layers.")[1].split(".")[0])
            if layer_num in keep_layers:
                new_key = key.replace(
                    f"layers.{layer_num}", f"layers.{keep_layers.index(layer_num)}"
                )
                new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # Update config to reflect reduced layers
    model.config.num_hidden_layers = len(keep_layers)

    # Load reduced state dict
    model.load_state_dict(new_state_dict)
    breakpoint()


def generate_response(messages):
    # Format the messages into DeepSeek's expected format
    formatted_prompt = ""
    for message in messages:
        if message["role"] == "user":
            formatted_prompt += f"Human: {message['content']}\n\nAssistant: "
        elif message["role"] == "assistant":
            formatted_prompt += f"{message['content']}\n\n"

    # Tokenize the input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode and return the response
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    return response


# Example usage
messages = [{"role": "user", "content": "Who are you?"}]
response = generate_response(messages)
print(response)
