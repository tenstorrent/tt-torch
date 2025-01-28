# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import os
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
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Modify config
    config.num_hidden_layers = 6
    config.num_attention_heads = 16
    config.hidden_size = 1024
    config.num_key_value_heads = 16
    config.intermediate_size = 1024 * 4
    config.num_experts_per_tok = 2
    config.q_lora_rank = 256

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create a new model with the config and all necessary parameters
    model = AutoModelForCausalLM.from_config(
        config,
        # device_map="cpu",  # Force CPU
        torch_dtype=torch.float32,  # Use float32
        attn_implementation="eager",  # Use eager implementation
        trust_remote_code=True,
    ).to(
        "cpu"
    )  # Ensure it's on CPU

    # Disable flash attention explicitly
    model.config.use_flash_attention = False
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
