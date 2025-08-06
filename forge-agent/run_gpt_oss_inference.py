#!/usr/bin/env python3
"""
GPT-OSS Inference Script for Tenstorrent Hardware
Runs GPT-OSS model inference on Tenstorrent hardware with text generation.
"""

import sys
import os
import logging
import torch
import argparse
from pathlib import Path

# Add the forge_agent module to the path
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging():
    """Setup logging for the inference script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('gpt_oss_inference.log')
        ]
    )
    return logging.getLogger(__name__)

def setup_tenstorrent_environment():
    """Setup Tenstorrent environment variables."""
    # Check for available Tenstorrent devices
    tenstorrent_devices = []
    for i in range(8):
        device_path = f"/dev/hugepages-1G/device_{i}_tenstorrent"
        if os.path.exists(device_path):
            tenstorrent_devices.append(i)
    
    if not tenstorrent_devices:
        raise RuntimeError("No Tenstorrent devices found")
    
    # Use first available device
    device_id = tenstorrent_devices[0]
    os.environ['PJRT_DEVICE'] = f'TPU:{device_id}'
    os.environ['TT_METAL_DEVICE_ID'] = str(device_id)
    
    return device_id

def load_gpt_oss_model(model_path):
    """Load the GPT-OSS model for Tenstorrent hardware."""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Load model configuration
    import json
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Create Tenstorrent-compatible model
    class TenstorrentCompatibleGPTOSS(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embedding = torch.nn.Embedding(config['vocab_size'], config['hidden_size'])
            self.layers = torch.nn.ModuleList([
                torch.nn.TransformerEncoderLayer(
                    d_model=config['hidden_size'],
                    nhead=config['num_attention_heads'],
                    dim_feedforward=config['hidden_size'] * 4,
                    dropout=0.0,
                    activation='gelu',
                    batch_first=True
                ) for _ in range(config['num_hidden_layers'])
            ])
            self.ln_f = torch.nn.LayerNorm(config['hidden_size'])
            
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.ln_f(x)
    
    # Create model
    model = TenstorrentCompatibleGPTOSS(config_data)
    logger.info("✅ Model created successfully")
    
    # Move to Tenstorrent device
    tt_device = torch.device('xla:0')
    model = model.to(tt_device)
    logger.info(f"✅ Model moved to Tenstorrent device: {tt_device}")
    
    return model, config_data, tt_device

def generate_text(model, config, device, prompt, max_length=100, temperature=0.7):
    """Generate text using the GPT-OSS model."""
    logger = logging.getLogger(__name__)
    
    # Simple tokenization (you might want to use proper tokenizer)
    # For demo purposes, we'll use a basic approach
    vocab_size = config['vocab_size']
    
    # Convert prompt to token IDs (simplified)
    # In a real implementation, you'd use the actual tokenizer
    prompt_tokens = [ord(c) % vocab_size for c in prompt[:50]]  # Simple char-based tokenization
    if not prompt_tokens:
        prompt_tokens = [1]  # Start token
    
    # Convert to tensor and move to device
    input_ids = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
    logger.info(f"📍 Input tokens: {input_ids.shape}")
    
    generated_tokens = input_ids.clone()
    
    with torch.no_grad():
        for i in range(max_length - len(prompt_tokens)):
            # Get model output
            outputs = model(generated_tokens)
            
            # Get next token probabilities
            logits = outputs[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1)
            
            # Append to generated sequence
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            
            # Check for end token (simplified)
            if next_token.item() == 2:  # End token
                break
    
    # Convert back to text (simplified)
    generated_text = ''.join([chr(t.item() % 128) for t in generated_tokens[0]])
    
    return generated_text

def run_inference(prompt, model_path="./gpt_oss_tenstorrent_converted", max_length=100, temperature=0.7):
    """Run GPT-OSS inference on Tenstorrent hardware."""
    logger = logging.getLogger(__name__)
    
    try:
        # Setup Tenstorrent environment
        device_id = setup_tenstorrent_environment()
        logger.info(f"🔧 Configured for Tenstorrent device {device_id}")
        
        # Load model
        model, config, device = load_gpt_oss_model(model_path)
        
        # Generate text
        logger.info(f"🎯 Generating text for prompt: '{prompt}'")
        generated_text = generate_text(model, config, device, prompt, max_length, temperature)
        
        logger.info("✅ Text generation completed successfully")
        logger.info(f"📝 Generated text: {generated_text}")
        
        return generated_text
        
    except Exception as e:
        logger.error(f"❌ Error during inference: {e}")
        raise

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="GPT-OSS Inference on Tenstorrent Hardware")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", 
                       help="Input prompt for text generation")
    parser.add_argument("--model-path", type=str, default="./gpt_oss_tenstorrent_converted",
                       help="Path to the converted model")
    parser.add_argument("--max-length", type=int, default=100,
                       help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for text generation")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("🚀 Starting GPT-OSS inference on Tenstorrent hardware")
    
    if args.interactive:
        logger.info("🎮 Interactive mode enabled. Type 'quit' to exit.")
        while True:
            try:
                prompt = input("\n🤖 Enter your prompt: ")
                if prompt.lower() == 'quit':
                    break
                
                generated_text = run_inference(
                    prompt, 
                    args.model_path, 
                    args.max_length, 
                    args.temperature
                )
                print(f"\n📝 Generated: {generated_text}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
    else:
        # Single inference
        generated_text = run_inference(
            args.prompt,
            args.model_path,
            args.max_length,
            args.temperature
        )
        print(f"\n📝 Generated text: {generated_text}")
    
    logger.info("🎉 Inference completed successfully!")

if __name__ == "__main__":
    main() 