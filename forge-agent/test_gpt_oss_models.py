#!/usr/bin/env python3
"""
Test script for OpenAI GPT-OSS models using the forge agent pipeline.
Tests the new open-source models: gpt-oss-20b and gpt-oss-120b
"""

import sys
import os
import logging
from pathlib import Path

# Add the forge_agent module to the path
sys.path.insert(0, str(Path(__file__).parent))

from forge_agent.pipeline import ModelCompatibilityPipeline
from forge_agent.metadata_service.models import ModelMetadata
from forge_agent.result_tracking.database import ResultTrackingDatabase

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('gpt_oss_test.log')
        ]
    )
    return logging.getLogger(__name__)

def test_gpt_oss_models():
    """Test the new OpenAI GPT-OSS models."""
    logger = setup_logging()
    logger.info("🚀 Starting GPT-OSS model testing")
    
    # Test the converted model directly
    converted_model_path = "./gpt_oss_tenstorrent_converted"
    
    if not os.path.exists(converted_model_path):
        logger.error(f"❌ Converted model not found at {converted_model_path}")
        logger.info("💡 Please run 'python adapt_gpt_oss_for_tenstorrent.py' first to convert the model")
        return
    
    logger.info(f"📁 Testing converted model at: {converted_model_path}")
    
    try:
        # Import and run the converted model test
        from test_converted_gpt_oss import test_converted_gpt_oss_model
        
        success = test_converted_gpt_oss_model()
        
        if success:
            logger.info("✅ GPT-OSS model testing completed successfully!")
        else:
            logger.error("❌ GPT-OSS model testing failed")
            
    except Exception as e:
        logger.error(f"❌ Error during testing: {str(e)}")
    
    logger.info("🏁 GPT-OSS model testing completed")

if __name__ == "__main__":
    test_gpt_oss_models() 