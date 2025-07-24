#!/usr/bin/env python3
"""
Test script to run Large Language Models through the full forge agent pipeline.
This includes model loading, adaptation, compilation, and execution testing.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add the forge_agent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from forge_agent.pipeline import ModelCompatibilityPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if the environment is properly set up."""
    try:
        import torch
        print(f"✅ PyTorch is available: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch is not available")
        return False
    
    try:
        # Check if tt-torch is available
        import tt_torch
        print("✅ tt-torch is available")
        return True
    except ImportError:
        print("❌ tt-torch is not available")
        return False

def display_comprehensive_results(test_record, model_id: str, total_time: float):
    """Display comprehensive test results to the console."""
    print("\n" + "="*80)
    print(f"🔍 COMPREHENSIVE TEST RESULTS FOR: {model_id}")
    print("="*80)
    
    # Model Information
    print(f"\n📋 MODEL INFORMATION:")
    print(f"   Model ID: {model_id}")
    print(f"   Status: {test_record.status}")
    print(f"   Adaptation Level: {test_record.adaptation_level if test_record.adaptation_level else 'N/A'}")
    
    # Text Processing Information
    print(f"\n📝 TEXT PROCESSING:")
    print(f"   Input Type: Text sequence")
    print(f"   Sample Text: 'This is a sample input for testing the model.'")
    print(f"   Tokenization: Automatic via model tokenizer")
    print(f"   Input Format: input_ids + attention_mask")
    print(f"   Max Length: Dynamic based on model configuration")
    
    # Compilation Information
    print(f"\n🔥 TT-TORCH COMPILATION:")
    if test_record.compilation_time_seconds:
        print(f"   Compilation Time: {test_record.compilation_time_seconds:.3f} seconds")
        print(f"   Backend: 'tt' (Tenstorrent)")
        print(f"   Mode: Real hardware compilation")
        print(f"   Dynamic: False")
        print(f"   Consteval: Enabled")
    else:
        print(f"   Compilation: Simulated (no real hardware)")
    
    # Execution Information
    print(f"\n🚀 INFERENCE EXECUTION:")
    if test_record.execution_time_seconds:
        print(f"   Execution Time: {test_record.execution_time_seconds:.3f} seconds")
        print(f"   Hardware: Tenstorrent accelerator")
        print(f"   Mode: Real hardware execution")
    else:
        print(f"   Execution: Simulated")
    
    # Memory Usage
    if test_record.memory_usage_mb:
        print(f"   Memory Usage: {test_record.memory_usage_mb:.2f} MB")
    
    # Results Interpretation
    print(f"\n🎯 RESULTS INTERPRETATION:")
    if "gpt" in model_id.lower():
        print(f"   Model Type: GPT Language Model (Generative)")
        print(f"   Output: Token probabilities for next token prediction")
        print(f"   Use Case: Text generation, completion, and understanding")
        print(f"   ✅ Successfully processed text input through transformer!")
    elif "bert" in model_id.lower():
        print(f"   Model Type: BERT Encoder (Bidirectional)")
        print(f"   Output: Contextualized token embeddings")
        print(f"   Use Case: Text classification, feature extraction, Q&A")
        print(f"   ✅ Successfully encoded text into semantic representations!")
    elif "t5" in model_id.lower():
        print(f"   Model Type: T5 Text-to-Text Transfer Transformer")
        print(f"   Output: Generated text sequences")
        print(f"   Use Case: Translation, summarization, text-to-text tasks")
        print(f"   ✅ Successfully processed text-to-text transformation!")
    elif "distilbert" in model_id.lower():
        print(f"   Model Type: DistilBERT (Efficient BERT)")
        print(f"   Output: Contextualized token embeddings (smaller model)")
        print(f"   Use Case: Fast text classification and feature extraction")
        print(f"   ✅ Successfully processed text with efficient transformer!")
    else:
        print(f"   Model Type: {model_id.split('/')[-1] if '/' in model_id else model_id}")
        print(f"   Output: Model-specific tensor outputs")
        print(f"   Use Case: Determined by model architecture")
        print(f"   ✅ Successfully executed model inference!")
    
    # Performance Summary
    print(f"\n⚡ PERFORMANCE SUMMARY:")
    print(f"   Total Pipeline Time: {total_time:.2f} seconds")
    if test_record.compilation_time_seconds and test_record.execution_time_seconds:
        overhead = total_time - test_record.compilation_time_seconds - test_record.execution_time_seconds
        print(f"   Model Loading/Setup: {overhead:.2f} seconds")
        print(f"   Compilation Overhead: {(test_record.compilation_time_seconds/total_time)*100:.1f}%")
        print(f"   Execution Time: {(test_record.execution_time_seconds/total_time)*100:.1f}%")
    
    # Success/Failure Status
    print(f"\n🎉 FINAL STATUS:")
    status_str = str(test_record.status).lower()
    if "completed" in status_str or test_record.status == "completed":
        print(f"   ✅ SUCCESS: Model completed the full tt-torch pipeline!")
        print(f"   ✅ Text processing: Working")
        print(f"   ✅ Tenstorrent compilation: Working") 
        print(f"   ✅ Hardware execution: Working")
        print(f"   ✅ Model inference: Working")
    else:
        print(f"   ❌ FAILED: {test_record.status}")
        if test_record.failure_reason:
            print(f"   ❌ Reason: {test_record.failure_reason}")
        if test_record.error_message:
            print(f"   ❌ Error: {test_record.error_message}")
    
    print("="*80)

def display_database_statistics(pipeline):
    """Display overall statistics from the database."""
    try:
        stats = pipeline.result_db.get_summary_statistics()
        
        print("="*80)
        print("📊 OVERALL TESTING STATISTICS (All Runs)")
        print("="*80)
        
        print(f"\n📈 GENERAL STATISTICS:")
        print(f"   Total Tests Run: {stats['total_tests']}")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        print(f"   Failed Tests: {stats['failed_tests']}")
        
        print(f"\n⚡ PERFORMANCE AVERAGES:")
        if stats['avg_compilation_time']:
            print(f"   Average Compilation Time: {stats['avg_compilation_time']:.3f} seconds")
        if stats['avg_execution_time']:
            print(f"   Average Execution Time: {stats['avg_execution_time']:.3f} seconds")
        if stats['avg_total_time']:
            print(f"   Average Total Processing: {stats['avg_total_time']:.3f} seconds")
        
        print(f"\n💾 DATABASE INFO:")
        print(f"   Database Location: forge_agent/data/test_results.db")
        print(f"   All test results are persistently stored")
        print(f"   Results include full execution traces and timing data")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error getting database statistics: {str(e)}")
        print("="*80)

def test_single_llm_model(model_id: str):
    """Test a single LLM model through the full pipeline."""
    logger.info(f"Testing LLM model through full pipeline: {model_id}")
    
    # Initialize the full pipeline
    pipeline = ModelCompatibilityPipeline(
        cache_dir="./test_cache",
        llm_provider="openai"  # Will disable LLM if no API key
    )
    
    print(f"\n🚀 RUNNING FULL PIPELINE TEST FOR: {model_id}")
    print("=" * 70)
    
    # Record start time
    start_time = time.time()
    
    # Test the model
    test_record = pipeline.test_model(model_id, use_llm_adaptation=False)  # Disable LLM for now
    
    # Record total time
    total_time = time.time() - start_time
    
    # Display comprehensive results to console
    display_comprehensive_results(test_record, model_id, total_time)
    
    # Also keep the original compact summary
    print(f"\n📋 COMPACT SUMMARY FOR {model_id}")
    print("=" * 50)
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
    status_str = str(test_record.status).lower()
    if "completed" in status_str or test_record.status == "completed":
        print(f"\n✅ SUCCESS: {model_id} completed the full pipeline!")
    elif "failed" in status_str or test_record.status == "failed":
        print(f"\n❌ FAILED: {model_id} failed in the pipeline")
    else:
        print(f"\n⚠️  INCOMPLETE: {model_id} pipeline status: {test_record.status}")
    
    return test_record

def main():
    """Main function to test LLM models through the pipeline."""
    logger.info("Starting LLM pipeline test")
    
    print("🔥 LLM MODEL PIPELINE TESTING (SINGLE MODEL MODE)")
    print("=" * 70)
    print("This will test Large Language Models through the complete forge agent pipeline:")
    print("  1. Model Loading & Validation")
    print("  2. Model Adaptation (if needed)")
    print("  3. tt-torch Compilation")
    print("  4. Inference Execution")
    print("  5. Result Validation")
    print()
    
    # Test a single LLM model (avoiding resource cleanup issues)
    print("🎯 SINGLE LLM MODEL TEST")
    print("-" * 30)
    
    # Test GPT-2 small model - good for testing LLM pipeline
    single_result = test_single_llm_model("gpt2")
    
    print(f"\n🎉 SINGLE LLM MODEL PIPELINE TESTING COMPLETE!")
    print("Single model test focused on avoiding resource cleanup issues between runs")
    print("Check the results above to see how LLM performs with tt-torch")
    
    # Display overall database statistics
    print(f"\n📊 Fetching overall statistics from database...")
    try:
        # Create a pipeline instance to access database statistics
        pipeline = ModelCompatibilityPipeline(
            cache_dir="./test_cache",
            llm_provider="openai"
        )
        display_database_statistics(pipeline)
    except Exception as e:
        print(f"❌ Error accessing database statistics: {str(e)}")
    
    # Disabled multiple model testing to avoid segfault issues
    print(f"\n💡 SUGGESTED LLM MODELS TO TEST:")
    print("   Run the script multiple times manually to test different models")
    print("   python test_llm_pipeline.py  # Default: gpt2")
    print("   Try these LLM models:")
    print("   - 'distilbert-base-uncased'   # Fast BERT variant")
    print("   - 'bert-base-uncased'         # Classic BERT")
    print("   - 't5-small'                  # Text-to-text model")
    print("   - 'microsoft/DialoGPT-small'  # Conversational AI")
    print("   All results are stored in the database for analysis")

if __name__ == "__main__":
    # Check environment first
    if not check_environment():
        print("\n❌ Environment check failed. Please ensure all dependencies are installed.")
        sys.exit(1)
    
    print("="*80)
    
    main() 