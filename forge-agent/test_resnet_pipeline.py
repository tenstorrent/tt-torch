#!/usr/bin/env python3
"""
Test script to run a ResNet model through the full forge agent pipeline.
This includes model loading, adaptation, compilation, and execution testing.
"""
import os
import sys

# Add the forge_agent module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from forge_agent.pipeline import ModelCompatibilityPipeline
from forge_agent.metadata_service.models import ModelSelectionCriteria, ModelFramework, ModelArchitecture
from forge_agent.test_pipeline.models import TestConfig
import time

def test_single_resnet_model(model_id: str):
    """Test a single ResNet model through the full pipeline."""
    logger.info(f"Testing ResNet model through full pipeline: {model_id}")
    
    # Initialize the full pipeline
    pipeline = ModelCompatibilityPipeline(
        cache_dir="./test_cache",
        llm_provider="openai"  # Will disable LLM if no API key
    )
    
    print(f"\n🚀 RUNNING FULL PIPELINE TEST FOR: {model_id}")
    print("=" * 70)
    
    # Record start time
    start_time = time.time()
    
    # Run the complete model test
    test_record = pipeline.test_model(model_id, use_llm_adaptation=False)  # Disable LLM for now
    
    # Record total time
    total_time = time.time() - start_time
    
    # Print detailed results
    print(f"\n📊 PIPELINE RESULTS FOR {model_id}")
    print("=" * 70)
    print(f"Status: {test_record.status}")
    print(f"Total Time: {total_time:.2f} seconds")
    
    if test_record.adaptation_level:
        print(f"Adaptation Level: {test_record.adaptation_level}")
    
    if test_record.compilation_time_seconds:
        print(f"Compilation Time: {test_record.compilation_time_seconds:.2f} seconds")
    
    if test_record.execution_time_seconds:
        print(f"Execution Time: {test_record.execution_time_seconds:.2f} seconds")
    
    if test_record.memory_usage_mb:
        print(f"Memory Usage: {test_record.memory_usage_mb:.2f} MB")
    
    if test_record.failure_reason:
        print(f"❌ Failure Reason: {test_record.failure_reason}")
    
    if test_record.error_message:
        print(f"❌ Error Message: {test_record.error_message}")
        
    if test_record.stack_trace:
        print(f"📋 Stack Trace:")
        print(test_record.stack_trace)
    
    # Success/failure indicator
    if test_record.status.value == "completed":
        print(f"\n✅ SUCCESS: {model_id} completed the full pipeline!")
    elif test_record.status.value == "failed":
        print(f"\n❌ FAILED: {model_id} failed in the pipeline")
    else:
        print(f"\n⚠️  INCOMPLETE: {model_id} pipeline status: {test_record.status}")
    
    return test_record

def test_multiple_resnet_models():
    """Test multiple ResNet models through the pipeline."""
    # List of successfully downloaded ResNet models
    resnet_models = [
        "microsoft/resnet-50",
        "microsoft/resnet-18", 
        "microsoft/resnet-34",
    ]
    
    print("🧪 TESTING MULTIPLE RESNET MODELS THROUGH PIPELINE")
    print("=" * 70)
    
    results = []
    
    for i, model_id in enumerate(resnet_models, 1):
        print(f"\n[{i}/{len(resnet_models)}] Testing: {model_id}")
        try:
            result = test_single_resnet_model(model_id)
            results.append((model_id, result))
        except Exception as e:
            logger.error(f"Error testing {model_id}: {str(e)}")
            print(f"❌ Exception occurred: {str(e)}")
            results.append((model_id, None))
    
    # Summary
    print(f"\n📋 SUMMARY OF ALL RESNET TESTS")
    print("=" * 70)
    
    successful = 0
    failed = 0
    
    for model_id, result in results:
        if result is None:
            print(f"❌ {model_id}: Exception occurred")
            failed += 1
        elif result.status.value == "completed":
            print(f"✅ {model_id}: SUCCESS ({result.compilation_time_seconds:.2f}s compile, {result.execution_time_seconds:.2f}s execute)")
            successful += 1
        else:
            print(f"❌ {model_id}: {result.status.value} - {result.failure_reason or 'Unknown'}")
            failed += 1
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"   Total models tested: {len(results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {successful/len(results)*100:.1f}%")
    
    return results

def main():
    """Main function to test ResNet models through the pipeline."""
    logger.info("Starting ResNet pipeline test")
    
    print("🔥 RESNET MODEL PIPELINE TESTING")
    print("=" * 70)
    print("This will test ResNet models through the complete forge agent pipeline:")
    print("  1. Model Loading & Validation")
    print("  2. Model Adaptation (if needed)")
    print("  3. tt-torch Compilation")
    print("  4. Inference Execution")
    print("  5. Result Validation")
    print()
    
    # Test a single model first
    print("🎯 SINGLE MODEL TEST")
    print("-" * 30)
    
    single_result = test_single_resnet_model("microsoft/resnet-50")
    
    # Ask if user wants to test multiple models
    print(f"\n🔄 MULTIPLE MODEL TEST")
    print("-" * 30)
    print("Testing multiple ResNet models...")
    
    # Test multiple models
    all_results = test_multiple_resnet_models()
    
    print(f"\n🎉 PIPELINE TESTING COMPLETE!")
    print("Check the results above to see how ResNet models perform with tt-torch")

if __name__ == "__main__":
    main() 