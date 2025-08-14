"""
Main entry point for the Hugging Face Model Compatibility Pipeline.
"""
import argparse
import os
import sys
import uvicorn
from loguru import logger
from typing import Optional
from urllib.parse import urlparse

from forge_agent.metadata_service.models import ModelSelectionCriteria, ModelFramework
from forge_agent.pipeline import ModelCompatibilityPipeline
from forge_agent.dashboard.api import app as dashboard_app


def setup_logging():
    """Set up logging configuration with human-readable formatting."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create subdirectories for different log types
    adaptation_log_dir = os.path.join(log_dir, "adaptations")
    os.makedirs(adaptation_log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "forge_agent.log")
    
    # Human-readable format for console
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Detailed format for file
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
    )
    
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level="INFO", format=console_format)  # Add console handler
    logger.add(log_file, rotation="10 MB", level="DEBUG", format=file_format)  # Add file handler
    
    logger.info(f"🚀 Forge Agent Pipeline Initialized")
    logger.info(f"📁 Main log file: {log_file}")
    logger.info(f"📂 Adaptation logs: {adaptation_log_dir}")


def _normalize_model_arg(model: str) -> str:
    """Accept either 'owner/model' or a full HF URL and return 'owner/model'."""
    if not model:
        return model
    if model.startswith("http://") or model.startswith("https://"):
        u = urlparse(model)
        # Paths can be '/owner/model' or '/models/owner/model'
        parts = [p for p in u.path.split('/') if p]
        if not parts:
            return model
        if parts[0] == 'models' and len(parts) >= 3:
            return f"{parts[1]}/{parts[2]}"
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return model


def _enum_or_str(value):
    """Return Enum.value if present, otherwise the original value.
    Helps when some fields may come back as plain strings instead of Enums.
    """
    try:
        return value.value  # Enum
    except Exception:
        return value

def run_pipeline(args):
    """Run the model compatibility pipeline."""
    # Create model selection criteria from args
    criteria = ModelSelectionCriteria(
        limit=args.limit,
        min_downloads=args.min_downloads,
        max_size_mb=args.max_size_mb
    )
    
    if args.pytorch_only:
        criteria.frameworks = [ModelFramework.PYTORCH]
    
    # Handle API keys
    setup_api_keys(args)
    
    # Initialize and run the pipeline
    pipeline = ModelCompatibilityPipeline(
        hf_api_token=os.environ.get("HF_API_TOKEN"),
        cache_dir=args.cache_dir,
        llm_provider=args.llm_provider
    )
    
    # If a specific model was requested, run only that model
    if getattr(args, 'model', None):
        model_id = _normalize_model_arg(args.model)
        logger.info(f"Running single-model test for: {model_id}")
        record = pipeline.test_model(model_id)
        status_val = _enum_or_str(record.status) if record.status is not None else None
        adaptation_val = _enum_or_str(record.adaptation_level) if getattr(record, 'adaptation_level', None) else None
        failure_val = _enum_or_str(record.failure_reason) if getattr(record, 'failure_reason', None) else None
        results = {
            "models_tested": 1,
            "successful_tests": 1 if status_val == 'completed' else 0,
            "failed_tests": 1 if status_val == 'failed' else 0,
            "model_id": record.model_id,
            "status": status_val,
            "adaptation_level": adaptation_val,
            "failure_reason": failure_val,
        }
    else:
        results = pipeline.run_pipeline(
            model_selection_criteria=criteria,
            max_concurrent=args.max_concurrent
        )
    
    # Print results
    logger.info(f"Pipeline completed. Results: {results}")
    
    return results


def run_dashboard(args):
    """Run the dashboard server."""
    logger.info(f"Starting dashboard server on http://localhost:{args.port}")
    uvicorn.run(dashboard_app, host=args.host, port=args.port)


def setup_api_keys(args):
    """Set up API keys from arguments or environment variables."""
    # Handle OpenAI API key
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
        logger.info("Using provided OpenAI API key")
    elif os.environ.get("OPENAI_API_KEY"):
        logger.info("Using OpenAI API key from environment variable")
        
    # Handle Anthropic API key
    if args.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic_api_key
        logger.info("Using provided Anthropic API key")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Using Anthropic API key from environment variable")
    
    # Check if selected LLM provider has an API key
    if args.llm_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OpenAI selected as LLM provider but no API key provided. LLM adaptation will be disabled.")
    elif args.llm_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("Anthropic selected as LLM provider but no API key provided. LLM adaptation will be disabled.")

def main():
    """Main entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Hugging Face Model Compatibility Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the compatibility pipeline")
    pipeline_parser.add_argument("--limit", type=int, default=10, 
                               help="Maximum number of models to test")
    pipeline_parser.add_argument("--min-downloads", type=int, default=None,
                               help="Minimum number of downloads")
    pipeline_parser.add_argument("--max-size-mb", type=float, default=None,
                               help="Maximum model size in MB")
    pipeline_parser.add_argument("--pytorch-only", action="store_true",
                               help="Only test PyTorch models")
    pipeline_parser.add_argument("--cache-dir", type=str, 
                               default=os.path.expanduser("~/.cache/huggingface/forge-agent"),
                               help="Cache directory for downloaded models")
    pipeline_parser.add_argument("--max-concurrent", type=int, default=1,
                               help="Maximum number of models to test concurrently")
    pipeline_parser.add_argument("--llm-provider", type=str, choices=["openai", "anthropic"], default="openai",
                               help="LLM provider to use for model adaptation")
    pipeline_parser.add_argument("--model", type=str, default=None,
                                help="Run pipeline on a specific Hugging Face model (repo ID like 'owner/model' or full URL)")
    pipeline_parser.add_argument("--openai-api-key", type=str, default=None,
                               help="OpenAI API key for LLM adaptation")
    pipeline_parser.add_argument("--anthropic-api-key", type=str, default=None,
                               help="Anthropic API key for LLM adaptation")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Run the dashboard server")
    dashboard_parser.add_argument("--host", type=str, default="127.0.0.1",
                                help="Host to run the dashboard server on")
    dashboard_parser.add_argument("--port", type=int, default=8000,
                                help="Port to run the dashboard server on")
    
    args = parser.parse_args()
    
    if args.command == "pipeline":
        run_pipeline(args)
    elif args.command == "dashboard":
        run_dashboard(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
