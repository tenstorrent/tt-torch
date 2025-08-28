# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import torch
import torch.nn as nn
import tt_torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import BackendOptions
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
from tt_torch.tools.device_manager import DeviceManager
import time

# Set Tenstorrent device for PJRT
os.environ['PJRT_DEVICE'] = 'TT'


# Clear dynamo cache function
def clear_dynamo_cache():
    # taken from/ inspired by: https://github.com/pytorch/pytorch/issues/107444
    import torch._dynamo as dynamo
    import gc
    dynamo.reset()  # clear cache
    gc.collect()


class SingleTransformerLayer(nn.Module):
    """Wrapper to extract and run a single transformer layer"""
    
    def __init__(self, layer, layer_norm=None):
        super().__init__()
        self.layer = layer
        self.layer_norm = layer_norm  # Optional pre-layer norm
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        
        # Call the transformer layer
        if hasattr(self.layer, '__call__'):
            # Most transformer layers expect these arguments
            layer_outputs = self.layer(
                hidden_states,
                attention_mask=attention_mask,
                **kwargs
            )
            # Handle different return types
            if isinstance(layer_outputs, tuple):
                return layer_outputs[0]  # Return just the hidden states
            else:
                return layer_outputs
        else:
            return self.layer(hidden_states)


@torch.inference_mode()
def main(use_test_model=False, layer_index=0, model_override=None):
    # Model selection - Default to large model similar to Grok
    if model_override:
        model_name = model_override
        print(f"Using specified model: {model_name}")
    elif use_test_model:
        model_name = "microsoft/DialoGPT-small"  # Small model for testing
        print(f"Using test model: {model_name}")
        print("Note: Using smaller model for testing single layer compilation.")
    else:
        model_name = "xai-org/grok-2"
        print(f"Loading Grok 2 model: {model_name}")
        print("Note: Loading actual Grok 2 model for single layer compilation.")

    try:
        # Load the full model first
        print("Loading full model...")
        if use_test_model:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                use_cache=False,  # Disable cache for single layer testing
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                use_cache=False,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        
        model = model.eval()
        
        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"Model loaded successfully. Total layers: {len(model.transformer.h) if hasattr(model, 'transformer') else 'Unknown'}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Extract the specified layer
    try:
        # Different models have different structures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Grok 2, LLaMA-style models - prioritize this for Grok 2
            layers = model.model.layers
            total_layers = len(layers)
            
            if layer_index >= total_layers:
                print(f"Error: Layer index {layer_index} >= total layers {total_layers}")
                return
                
            target_layer = layers[layer_index]
            layer_norm = None
            print(f"Detected Grok/LLaMA-style architecture")
            
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-style models (DialoGPT, GPT-2, etc.)
            layers = model.transformer.h
            total_layers = len(layers)
            
            if layer_index >= total_layers:
                print(f"Error: Layer index {layer_index} >= total layers {total_layers}")
                return
                
            target_layer = layers[layer_index]
            
            # Also get layer norm if it exists
            layer_norm = None
            if hasattr(model.transformer, 'ln_f'):
                layer_norm = model.transformer.ln_f
            print(f"Detected GPT-style architecture")
            
        else:
            print("Error: Unknown model architecture")
            print(f"Available attributes: {dir(model)}")
            return
            
        print(f"Extracted layer {layer_index} of {total_layers}")
        print(f"Layer type: {type(target_layer)}")
        
        # Create single layer wrapper
        single_layer = SingleTransformerLayer(target_layer, layer_norm)
        
    except Exception as e:
        print(f"Error extracting layer: {e}")
        return

    # Configure tt-torch compiler
    clear_dynamo_cache()
    cc = CompilerConfig()
    cc.enable_consteval = False
    cc.consteval_parameters = False

    options = BackendOptions()
    options.compiler_config = cc

    # Create device mesh (single device for single layer)
    try:
        device = DeviceManager.create_parent_mesh_device(mesh_shape=[1, 1])
        print("Using single device for single layer...")
        options.devices = [device]
    except Exception as e:
        print(f"Warning: Could not create device mesh: {e}")
        print("Continuing without device mesh...")
        device = None

    # Compile the single layer with tt-torch backend
    print("Compiling single layer with tt-torch backend...")
    try:
        compiled_layer = torch.compile(single_layer, backend="tt", dynamic=False, options=options)
        print("Single layer compilation successful!")
    except Exception as e:
        print(f"Layer compilation failed: {e}")
        if device:
            DeviceManager.release_parent_device(device)
        return

    # Test the compiled layer
    print("\nTesting compiled layer...")
    
    # Create test input
    test_prompt = "Testing single layer:"
    inputs = tokenizer.encode_plus(
        test_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=64,  # Small sequence for testing
        return_attention_mask=True,
    )
    
    # Get embeddings from the full model
    with torch.no_grad():
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            # Grok 2, LLaMA-style embedding - prioritize for Grok 2
            embeddings = model.model.embed_tokens(inputs.input_ids).to(torch.bfloat16)
            print("Using Grok/LLaMA-style embeddings")
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            # GPT-style embedding
            embeddings = model.transformer.wte(inputs.input_ids).to(torch.bfloat16)
            print("Using GPT-style embeddings")
        else:
            print("Error: Could not find embedding layer")
            print("Available model attributes:", [attr for attr in dir(model) if not attr.startswith('_')])
            if device:
                DeviceManager.release_parent_device(device)
            return
    
    print(f"Input shape: {embeddings.shape}")
    print(f"Input prompt: '{test_prompt}'")
    
    # Time the layer execution
    num_runs = 3
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        
        try:
            # Run the compiled single layer
            with torch.no_grad():
                layer_output = compiled_layer(
                    embeddings,
                    attention_mask=inputs.attention_mask.to(torch.bfloat16)
                )
            
            end_time = time.time()
            run_time = end_time - start_time
            times.append(run_time)
            
            print(f"Run {i+1}: {run_time:.4f}s, Output shape: {layer_output.shape}")
            
        except Exception as e:
            print(f"Error during layer execution run {i+1}: {e}")
            break
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\nAverage execution time: {avg_time:.4f}s")
        print(f"Layer {layer_index} successfully processed on tt-torch backend!")
        
        # Show some output statistics
        print(f"Output statistics:")
        print(f"  Mean: {layer_output.mean().item():.6f}")
        print(f"  Std: {layer_output.std().item():.6f}")
        print(f"  Min: {layer_output.min().item():.6f}")
        print(f"  Max: {layer_output.max().item():.6f}")

    # Clean up
    print("\nCleaning up...")
    if device:
        DeviceManager.release_parent_device(device)
    clear_dynamo_cache()
    print("Single layer demo completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Layer Demo for tt-torch compilation")
    parser.add_argument(
        "--test_model",
        action="store_true",
        help="Use DialoGPT-small test model instead of Grok 2 (for testing).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Layer index to extract and compile (default: 0 - first layer).",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specify exact model name (e.g. 'xai-org/grok-2' if you have access).",
    )
    args = parser.parse_args()
    
    use_test_model = args.test_model
    main(use_test_model, args.layer, args.model)
