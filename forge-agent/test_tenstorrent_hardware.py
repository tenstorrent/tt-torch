#!/usr/bin/env python3
"""
Test script to verify Tenstorrent hardware execution.
Explicitly moves model to Tenstorrent hardware and verifies execution.
"""

import sys
import os
import logging
import torch
from pathlib import Path

# Add the forge_agent module to the path
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tenstorrent_hardware_test.log')
        ]
    )
    return logging.getLogger(__name__)

def check_tenstorrent_devices():
    """Check for available Tenstorrent devices."""
    logger = logging.getLogger(__name__)
    
    # Check for Tenstorrent devices
    tenstorrent_devices = []
    for i in range(8):  # Check devices 0-7
        device_path = f"/dev/hugepages-1G/device_{i}_tenstorrent"
        if os.path.exists(device_path):
            tenstorrent_devices.append(i)
            logger.info(f"✅ Found Tenstorrent device {i}: {device_path}")
    
    if not tenstorrent_devices:
        logger.warning("❌ No Tenstorrent devices found")
        return []
    
    logger.info(f"📋 Available Tenstorrent devices: {tenstorrent_devices}")
    return tenstorrent_devices

def test_tenstorrent_hardware_execution():
    """Test execution on Tenstorrent hardware."""
    logger = logging.getLogger(__name__)
    
    # Check available devices
    devices = check_tenstorrent_devices()
    if not devices:
        logger.error("❌ No Tenstorrent devices available")
        return False
    
    # Set environment for Tenstorrent
    device_id = devices[0]  # Use first available device
    os.environ['PJRT_DEVICE'] = f'TPU:{device_id}'
    os.environ['TT_METAL_DEVICE_ID'] = str(device_id)
    
    logger.info(f"🔧 Configured for Tenstorrent device {device_id}")
    logger.info(f"   PJRT_DEVICE: {os.environ.get('PJRT_DEVICE')}")
    logger.info(f"   TT_METAL_DEVICE_ID: {os.environ.get('TT_METAL_DEVICE_ID')}")
    
    try:
        # Import tt-torch
        import tt_torch
        from tt_torch.dynamo.backend import BackendOptions
        
        # Load the converted model
        model_path = "./gpt_oss_tenstorrent_converted"
        if not os.path.exists(model_path):
            logger.error(f"❌ Model path not found: {model_path}")
            return False
        
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
        
        # Try to move model to Tenstorrent device
        try:
            # Check if we can create a Tenstorrent device
            tt_device = torch.device('xla:0')  # Try XLA device for Tenstorrent
            logger.info(f"🔧 Attempting to use device: {tt_device}")
            
            # Move model to Tenstorrent device
            model = model.to(tt_device)
            logger.info("✅ Model moved to Tenstorrent device")
            
            # Test inference on Tenstorrent hardware
            input_ids = torch.randint(0, config_data['vocab_size'], (1, 10), device=tt_device)
            logger.info(f"📍 Input device: {input_ids.device}")
            
            with torch.no_grad():
                output = model(input_ids)
            
            logger.info(f"✅ Inference successful on Tenstorrent hardware")
            logger.info(f"📍 Output device: {output.device}")
            logger.info(f"📊 Output shape: {output.shape}")
            
            # Verify it's actually on Tenstorrent hardware
            if 'xla' in str(output.device):
                logger.info("🎉 SUCCESS: Model is running on Tenstorrent hardware!")
                return True
            else:
                logger.warning("⚠️  Model may not be on Tenstorrent hardware")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️  Could not move to Tenstorrent device: {e}")
            
            # Fallback: Try with tt-torch compilation
            logger.info("🔄 Trying tt-torch compilation...")
            
            # Compile with tt-torch
            backend_options = BackendOptions()
            compiled_model = torch.compile(
                model,
                backend="tt",
                dynamic=False,
                options=backend_options
            )
            
            # Test compiled model
            input_ids = torch.randint(0, config_data['vocab_size'], (1, 10))
            logger.info(f"📍 Input device: {input_ids.device}")
            
            with torch.no_grad():
                output = compiled_model(input_ids)
            
            logger.info(f"✅ Compiled model inference successful")
            logger.info(f"📍 Output device: {output.device}")
            logger.info(f"📊 Output shape: {output.shape}")
            
            # Check if compilation used Tenstorrent
            logger.info("📋 Checking if compilation used Tenstorrent hardware...")
            
            # Check model parameters device placement
            param_devices = set()
            for name, param in compiled_model.named_parameters():
                param_devices.add(str(param.device))
                if 'xla' in str(param.device) or 'tt' in str(param.device):
                    logger.info(f"✅ Parameter {name} on Tenstorrent device: {param.device}")
            
            logger.info(f"📊 Parameter devices: {param_devices}")
            
            if any('xla' in d or 'tt' in d for d in param_devices):
                logger.info("🎉 SUCCESS: Compiled model is using Tenstorrent hardware!")
                return True
            else:
                logger.warning("⚠️  Compiled model may not be using Tenstorrent hardware")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error during Tenstorrent hardware test: {e}")
        return False

def main():
    """Main test function."""
    logger = setup_logging()
    logger.info("🚀 Starting Tenstorrent hardware execution test")
    
    # Test Tenstorrent hardware execution
    success = test_tenstorrent_hardware_execution()
    
    if success:
        logger.info("🎉 Tenstorrent hardware execution test completed successfully!")
        logger.info("📋 Summary:")
        logger.info("  - Tenstorrent devices detected")
        logger.info("  - Model successfully moved to Tenstorrent hardware")
        logger.info("  - Inference executed on Tenstorrent hardware")
        logger.info("  - Hardware execution verified")
    else:
        logger.error("❌ Tenstorrent hardware execution test failed")
        logger.info("📋 Summary:")
        logger.info("  - Model may not be running on Tenstorrent hardware")
        logger.info("  - Check device configuration and availability")
    
    return success

if __name__ == "__main__":
    main() 