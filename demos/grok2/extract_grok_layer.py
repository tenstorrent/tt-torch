# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import torch
import safetensors
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoConfig
import json

# Set Tenstorrent device for PJRT
os.environ['PJRT_DEVICE'] = 'TT'

def download_grok_layer(layer_index=0, local_dir="./grok_weights"):
    """Download and extract a specific layer from Grok 2"""
    
    model_name = "xai-org/grok-2"
    
    print(f"Attempting to download Grok 2 layer {layer_index}...")
    
    try:
        # First, get the config to understand the model structure
        print("Downloading model config...")
        config_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Model config loaded:")
        print(f"  Hidden size: {config.get('hidden_size', 'unknown')}")
        print(f"  Number of layers: {config.get('num_hidden_layers', 'unknown')}")
        print(f"  Number of attention heads: {config.get('num_attention_heads', 'unknown')}")
        
        # List available files
        print("\nListing available files...")
        files = list_repo_files(model_name)
        safetensor_files = [f for f in files if f.endswith('.safetensors')]
        
        print(f"Found {len(safetensor_files)} safetensor files:")
        for f in safetensor_files[:5]:  # Show first 5
            print(f"  {f}")
        if len(safetensor_files) > 5:
            print(f"  ... and {len(safetensor_files) - 5} more")
        
        # Try to download the first safetensor file to see the structure
        if safetensor_files:
            print(f"\nDownloading first safetensor file: {safetensor_files[0]}")
            first_file_path = hf_hub_download(
                repo_id=model_name,
                filename=safetensor_files[0],
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            
            # Examine the keys in the safetensor file
            print("Examining tensor keys...")
            metadata = safetensors.torch.safe_open(first_file_path, framework="pt", device="cpu")
            keys = list(metadata.keys())
            
            print(f"Found {len(keys)} tensors in first file:")
            layer_keys = [k for k in keys if f"layers.{layer_index}." in k]
            
            if layer_keys:
                print(f"Layer {layer_index} tensors found:")
                for key in layer_keys[:10]:  # Show first 10
                    tensor_shape = metadata.get_tensor(key).shape
                    print(f"  {key}: {tensor_shape}")
                
                return True, layer_keys, first_file_path
            else:
                print(f"No tensors found for layer {layer_index} in first file")
                print("Available layer patterns:")
                layer_patterns = set()
                for key in keys:
                    if "layers." in key:
                        layer_part = key.split("layers.")[1].split(".")[0]
                        layer_patterns.add(layer_part)
                print(f"  Found layers: {sorted(layer_patterns)}")
                
                return False, [], first_file_path
        else:
            print("No safetensor files found!")
            return False, [], None
            
    except Exception as e:
        print(f"Error downloading Grok 2: {e}")
        return False, [], None

def create_minimal_grok_layer(layer_keys, safetensor_path, layer_index=0):
    """Create a minimal layer from the downloaded weights"""
    
    print(f"\nCreating minimal layer {layer_index}...")
    
    try:
        # Load the specific layer tensors
        layer_weights = {}
        
        with safetensors.torch.safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for key in layer_keys:
                tensor = f.get_tensor(key)
                layer_weights[key] = tensor
                print(f"Loaded {key}: {tensor.shape}")
        
        print(f"Successfully loaded {len(layer_weights)} tensors for layer {layer_index}")
        return layer_weights
        
    except Exception as e:
        print(f"Error creating minimal layer: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract single layer from Grok 2")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to extract")
    args = parser.parse_args()
    
    print("=== Grok 2 Layer Extraction Tool ===")
    print(f"Target layer: {args.layer}")
    print()
    
    success, layer_keys, file_path = download_grok_layer(args.layer)
    
    if success:
        weights = create_minimal_grok_layer(layer_keys, file_path, args.layer)
        if weights:
            print(f"\n✅ SUCCESS: Layer {args.layer} extracted successfully!")
            print(f"Memory usage: ~{sum(w.numel() * w.element_size() for w in weights.values()) / 1024**3:.2f} GB")
        else:
            print(f"\n❌ FAILED: Could not create layer {args.layer}")
    else:
        print(f"\n❌ FAILED: Could not download layer {args.layer}")
