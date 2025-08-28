# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import torch
import tt_torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import BackendOptions
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
)
from tt_torch.tools.device_manager import DeviceManager
# Clear dynamo cache function
def clear_dynamo_cache():
    # taken from/ inspired by: https://github.com/pytorch/pytorch/issues/107444
    import torch._dynamo as dynamo
    import gc
    dynamo.reset()  # clear cache
    gc.collect()


@torch.inference_mode()
def main(run_interactive, use_test_model=False):
    # Grok 2 model from xAI on Hugging Face
    if use_test_model:
        # Use a smaller model for testing the tt-torch compilation pipeline
        model_name = "microsoft/DialoGPT-small"  # Much smaller for testing
        print(f"Using test model: {model_name}")
        print("Note: Using smaller model for testing tt-torch compilation pipeline.")
    else:
        model_name = "xai-org/grok-2"
        print(f"Loading Grok 2 model: {model_name}")
        print("Note: This is a very large model (~500GB). Make sure you have sufficient memory and storage.")
    
    try:
        # Load the model with appropriate settings
        if use_test_model:
            # Simpler loading for test model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                use_cache=True,
            )
        else:
            # Full Grok 2 loading with device mapping
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                use_cache=True,
                device_map="auto",  # Automatically distribute across available GPUs
                trust_remote_code=True,  # Required for some custom model architectures
            )
        model = model.eval()
        
        # Load tokenizer
        if use_test_model:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        
        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"Error loading Grok 2 model: {e}")
        print("This could be due to:")
        print("1. Insufficient memory/storage (model is ~500GB)")
        print("2. Missing authentication for Hugging Face")
        print("3. Network connectivity issues")
        print("Please ensure you have:")
        print("- At least 500GB free storage")
        print("- Multiple GPUs with >40GB VRAM each")
        print("- Valid Hugging Face authentication if required")
        return

    # Configure tt-torch compiler
    clear_dynamo_cache()
    cc = CompilerConfig()
    cc.enable_consteval = False  # Disable for large models to avoid memory issues
    cc.consteval_parameters = False

    options = BackendOptions()
    options.compiler_config = cc

    # Create device mesh
    try:
        if use_test_model:
            # Single device for test model
            device = DeviceManager.create_parent_mesh_device(mesh_shape=[1, 1])
            print("Using single device for test model...")
        else:
            # 8 GPUs for Grok 2
            device = DeviceManager.create_parent_mesh_device(mesh_shape=[1, 8])
            print("Using 8-device mesh for Grok 2...")
        options.devices = [device]
    except Exception as e:
        print(f"Warning: Could not create device mesh: {e}")
        print("Falling back to single device...")
        device = DeviceManager.create_parent_mesh_device(mesh_shape=[1, 1])
        options.devices = [device]

    # Compile the model with tt-torch backend
    print("Compiling model with tt-torch backend...")
    try:
        tt_model = torch.compile(model, backend="tt", dynamic=False, options=options)
        print("Model compilation successful!")
    except Exception as e:
        print(f"Model compilation failed: {e}")
        DeviceManager.release_parent_device(device)
        return

    # Set up text streamer for real-time output
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    def generate_text(prompt):
        print(f"\nPrompt: {prompt}")
        print("Grok response: ", end="")
        
        # Encode the input prompt
        inputs = tokenizer.encode_plus(
            prompt,
            return_tensors="pt",
            truncation=True,
            return_attention_mask=True,
        )

        # Generation parameters
        generation_args = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": 100,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        try:
            # Generate response using the compiled model
            outputs = tt_model.generate(**generation_args, streamer=streamer)
            print("\n" + "="*50)
        except Exception as e:
            print(f"\nGeneration failed: {e}")

    # Default prompts to demonstrate Grok's capabilities
    default_prompts = [
        "What is the meaning of life?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the key differences between machine learning and deep learning?",
    ]

    if not run_interactive:
        print("\nRunning with default prompts...")
        for prompt in default_prompts:
            generate_text(prompt)
    else:
        print("\nInteractive mode - you can chat with Grok 2!")
        print('Type "quit" or "exit" to stop.')
        
        while True:
            user_input = input("\nEnter your prompt: ").strip()
            if user_input.lower() in ["quit", "exit", "stop"]:
                break
            if user_input:
                generate_text(user_input)

    # Clean up
    print("\nCleaning up...")
    DeviceManager.release_parent_device(device)
    clear_dynamo_cache()
    print("Demo completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grok 2 Demo with tt-torch compilation")
    parser.add_argument(
        "--run_interactive",
        action="store_true",
        help="Run the demo interactively to chat with Grok 2.",
    )
    parser.add_argument(
        "--test_model",
        action="store_true",
        help="Use a smaller test model instead of Grok 2 for testing the compilation pipeline.",
    )
    args = parser.parse_args()
    main(args.run_interactive, args.test_model)
