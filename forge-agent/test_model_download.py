#!/usr/bin/env python3
"""
Simple test script to test model selection and downloading functionality.
Focus on getting top 5 popular models and testing the download process.
"""
import os
import sys

# Add the forge_agent module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from forge_agent.metadata_service.huggingface_client import HuggingFaceClient
from forge_agent.metadata_service.models import ModelSelectionCriteria, ModelFramework, ModelArchitecture
from forge_agent.test_pipeline.downloader import ModelDownloader
from forge_agent.test_pipeline.models import TestConfig

def main():
    """Test model selection and downloading."""
    logger.info("Starting model selection and download test")
    
    # Initialize components
    hf_client = HuggingFaceClient()
    downloader = ModelDownloader(cache_dir="./test_cache")
    
    # Get top models from Hugging Face
    logger.info("Fetching top 20 models from Hugging Face to find ResNet models...")
    try:
        top_models = hf_client.get_top_models(limit=20)
        logger.info(f"Successfully fetched {len(top_models)} models")
        
        # Print top 20 models with their details
        print("\n=== TOP 20 MODELS FROM HUGGING FACE ===")
        for i, model in enumerate(top_models[:20], 1):
            downloads = model.get("downloads", 0)
            model_id = model.get("id", "unknown")
            print(f"{i:2d}. {model_id:40s} Downloads: {downloads:,}")
        
        # Look for ResNet models specifically
        resnet_models = []
        pytorch_models = []
        
        print("\n=== ANALYZING MODELS FOR RESNET AND PYTORCH ===")
        for model in top_models:
            model_id = model.get("id", "").lower()
            
            # Check for ResNet in the model ID
            if "resnet" in model_id:
                resnet_models.append(model)
                print(f"🎯 Found ResNet model: {model.get('id')} (Downloads: {model.get('downloads', 0):,})")
            
            # Check for PyTorch models
            tags = model.get("tags", [])
            library = model.get("library_name", "").lower()
            if "pytorch" in library or "torch" in library or any("pytorch" in str(tag).lower() for tag in tags):
                pytorch_models.append(model)
        
        print(f"\nFound {len(resnet_models)} ResNet models and {len(pytorch_models)} PyTorch models")
        
        # Test downloading top 5 models
        test_models = top_models[:5]
        print(f"\n=== TESTING DOWNLOAD FOR TOP 5 MODELS ===")
        
        successful_downloads = 0
        for i, model in enumerate(test_models, 1):
            model_id = model.get("id")
            if not model_id:
                continue
                
            print(f"\n{i}. Testing download for: {model_id}")
            
            # Get detailed metadata
            try:
                metadata = hf_client.get_model_metadata(model_id)
                print(f"   Framework: {metadata.framework}")
                print(f"   Architecture: {metadata.architecture}")
                print(f"   Downloads: {metadata.downloads:,}")
                print(f"   Size: {metadata.size_mb:.2f} MB" if metadata.size_mb else "   Size: Unknown")
                
                # Test download
                test_config = TestConfig(model_id=model_id)
                success, model_data, failure_reason, error_msg = downloader.download_model(test_config)
                
                if success:
                    print(f"   ✅ Download successful!")
                    if model_data and "model" in model_data:
                        print(f"   Model type: {type(model_data['model'])}")
                    successful_downloads += 1
                else:
                    print(f"   ❌ Download failed: {failure_reason}")
                    if error_msg:
                        print(f"   Error: {error_msg}")
                        
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        print(f"\n=== SUMMARY ===")
        print(f"Total models tested: {len(test_models)}")
        print(f"Successful downloads: {successful_downloads}")
        print(f"ResNet models found: {len(resnet_models)}")
        print(f"PyTorch models found: {len(pytorch_models)}")
        
        # If we found ResNet models, try to download one
        if resnet_models:
            print(f"\n=== TESTING RESNET MODEL DOWNLOAD ===")
            resnet_model = resnet_models[0]
            model_id = resnet_model.get("id")
            print(f"Testing ResNet model: {model_id}")
            
            metadata = hf_client.get_model_metadata(model_id)
            print(f"Framework: {metadata.framework}")
            print(f"Architecture: {metadata.architecture}")
            
            test_config = TestConfig(model_id=model_id)
            success, model_data, failure_reason, error_msg = downloader.download_model(test_config)
            
            if success:
                print("✅ ResNet model download successful!")
                if model_data and "model" in model_data:
                    print(f"Model type: {type(model_data['model'])}")
            else:
                print(f"❌ ResNet model download failed: {failure_reason}")
                if error_msg:
                    print(f"Error: {error_msg}")
        
    except Exception as e:
        logger.error(f"Error in model selection/download test: {str(e)}")
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    main() 