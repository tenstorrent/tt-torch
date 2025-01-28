# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import os
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_imports


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-V3"
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    breakpoint()
    # Load the model for CPU, ensuring no flash-attn and GPU-related optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cpu",  # Explicitly force the model to run on CPU
        torch_dtype=torch.float32,  # Use float32 on CPU
        attn_implementation="eager",
    )

    # Ensure that we avoid any specific GPU-accelerated attention mechanisms like flash-attn
    model.config.use_flash_attention = False  # This may not exist in every model, but try to turn off flash-attn if possible


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
