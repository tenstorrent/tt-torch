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
    
    # Initialize the pipeline
    pipeline = ModelCompatibilityPipeline(
        llm_provider="anthropic",
        cache_dir="./gpt_oss_cache"
    )
    
    # Define the GPT-OSS models to test (only the smaller one for now)
    gpt_oss_models = [
        "openai/gpt-oss-20b"
    ]
    
    logger.info(f"📋 Testing {len(gpt_oss_models)} GPT-OSS models")
    
    for model_id in gpt_oss_models:
        logger.info(f"🧪 Testing model: {model_id}")
        
        try:
            # Test the model using the pipeline with LLM adaptation
            test_record = pipeline.test_model(model_id, use_llm_adaptation=True)
            
            if test_record.status == "completed" and test_record.result == "success":
                logger.info(f"✅ Successfully tested {model_id}")
            else:
                logger.error(f"❌ Failed to test {model_id}: {test_record.error_message}")
                
        except Exception as e:
            logger.error(f"❌ Error testing {model_id}: {str(e)}")
    
    logger.info("🏁 GPT-OSS model testing completed")

if __name__ == "__main__":
    test_gpt_oss_models() 