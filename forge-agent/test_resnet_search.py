#!/usr/bin/env python3
"""
Test script to specifically search for ResNet models and test downloads.
"""
import os
import sys

# Add the forge_agent module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from forge_agent.metadata_service.huggingface_client import HuggingFaceClient
from forge_agent.test_pipeline.downloader import ModelDownloader
from forge_agent.test_pipeline.models import TestConfig
import requests

def search_resnet_models():
    """Search specifically for ResNet models on Hugging Face."""
    logger.info("Searching for ResNet models on Hugging Face")
    
    # Search using HF API with query
    base_url = "https://huggingface.co/api/models"
    
    # Try different search terms for ResNet
    search_terms = ["resnet", "ResNet", "resnet50", "resnet18", "resnet34", "resnet101"]
    all_models = []
    
    for term in search_terms:
        logger.info(f"Searching for models with term: {term}")
        
        params = {
            "search": term,
            "limit": 20,
            "sort": "downloads",
            "direction": -1  # Descending order
        }
        
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                models = response.json()
                logger.info(f"Found {len(models)} models for term '{term}'")
                
                for model in models:
                    model_id = model.get("id", "")
                    downloads = model.get("downloads", 0)
                    # Only add if not already in our list
                    if not any(m.get("id") == model_id for m in all_models):
                        all_models.append(model)
                        print(f"  Found: {model_id:50s} Downloads: {downloads:,}")
            else:
                logger.error(f"Search failed for term '{term}': {response.status_code}")
        except Exception as e:
            logger.error(f"Error searching for '{term}': {str(e)}")
    
    # Sort by downloads and return top models
    all_models.sort(key=lambda x: x.get("downloads", 0), reverse=True)
    return all_models

def test_timm_resnet_models():
    """Test some popular ResNet models from timm library."""
    # Some known popular ResNet model IDs from timm/torchvision
    known_resnet_models = [
        "microsoft/resnet-50",
        "microsoft/resnet-18", 
        "microsoft/resnet-34",
        "microsoft/resnet-101",
        "microsoft/resnet-152",
        "timm/resnet50.a1_in1k",
        "timm/resnet18.a1_in1k",
        "pytorch/vision/resnet50",
        "huggingface/pytorch-image-models", # This contains ResNet models
    ]
    
    hf_client = HuggingFaceClient()
    downloader = ModelDownloader(cache_dir="./test_cache")
    
    print("\n=== TESTING KNOWN RESNET MODEL IDs ===")
    successful_models = []
    
    for model_id in known_resnet_models:
        print(f"\nTesting: {model_id}")
        
        try:
            # Get metadata first
            metadata = hf_client.get_model_metadata(model_id)
            print(f"  Framework: {metadata.framework}")
            print(f"  Architecture: {metadata.architecture}")
            print(f"  Downloads: {metadata.downloads:,}")
            
            # Test download
            test_config = TestConfig(model_id=model_id)
            success, model_data, failure_reason, error_msg = downloader.download_model(test_config)
            
            if success:
                print(f"  ✅ Download successful!")
                if model_data and "model" in model_data:
                    print(f"  Model type: {type(model_data['model'])}")
                successful_models.append(model_id)
            else:
                print(f"  ❌ Download failed: {failure_reason}")
                if error_msg:
                    print(f"  Error: {error_msg}")
                    
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    return successful_models

def main():
    """Main function to search and test ResNet models."""
    logger.info("Starting ResNet model search and test")
    
    # Search for ResNet models
    print("=== SEARCHING FOR RESNET MODELS ===")
    resnet_models = search_resnet_models()
    
    print(f"\n=== FOUND {len(resnet_models)} RESNET MODELS ===")
    
    if resnet_models:
        # Initialize components
        hf_client = HuggingFaceClient()
        downloader = ModelDownloader(cache_dir="./test_cache")
        
        # Test top 5 ResNet models
        test_models = resnet_models[:5]
        print(f"\n=== TESTING TOP 5 RESNET MODELS ===")
        
        successful_downloads = 0
        for i, model in enumerate(test_models, 1):
            model_id = model.get("id")
            downloads = model.get("downloads", 0)
            
            print(f"\n{i}. Testing ResNet model: {model_id} (Downloads: {downloads:,})")
            
            try:
                # Get detailed metadata
                metadata = hf_client.get_model_metadata(model_id)
                print(f"   Framework: {metadata.framework}")
                print(f"   Architecture: {metadata.architecture}")
                print(f"   Size: {metadata.size_mb:.2f} MB" if metadata.size_mb else "   Size: Unknown")
                
                # Test download
                test_config = TestConfig(model_id=model_id)
                success, model_data, failure_reason, error_msg = downloader.download_model(test_config)
                
                if success:
                    print(f"   ✅ Download successful!")
                    if model_data and "model" in model_data:
                        print(f"   Model type: {type(model_data['model'])}")
                        
                        # Print model architecture info if available
                        model_obj = model_data["model"]
                        if hasattr(model_obj, 'config'):
                            print(f"   Model config: {type(model_obj.config)}")
                        
                    successful_downloads += 1
                else:
                    print(f"   ❌ Download failed: {failure_reason}")
                    if error_msg:
                        print(f"   Error: {error_msg}")
                        
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        print(f"\n=== RESNET SEARCH SUMMARY ===")
        print(f"ResNet models found: {len(resnet_models)}")
        print(f"Models tested: {len(test_models)}")
        print(f"Successful downloads: {successful_downloads}")
    
    # Also test some known ResNet models
    successful_known = test_timm_resnet_models()
    
    print(f"\n=== OVERALL SUMMARY ===")
    print(f"Search-found ResNet models: {len(resnet_models) if resnet_models else 0}")
    print(f"Successfully downloaded known models: {len(successful_known)}")
    
    if successful_known:
        print("✅ Successfully downloaded ResNet models:")
        for model_id in successful_known:
            print(f"  - {model_id}")

if __name__ == "__main__":
    main() 