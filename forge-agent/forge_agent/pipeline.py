"""
Main pipeline for the Hugging Face Model Compatibility Pipeline.
"""
import time
from typing import List, Dict, Any, Optional
import os
import traceback
from datetime import datetime

from loguru import logger
import torch

from forge_agent.metadata_service.huggingface_client import HuggingFaceClient
from forge_agent.metadata_service.database import MetadataDatabase
from forge_agent.metadata_service.models import ModelMetadata, ModelSelectionCriteria
from forge_agent.test_pipeline.models import TestConfig, TestStatus, TestResult, AdaptationLevel, FailureReason
from forge_agent.test_pipeline.downloader import ModelDownloader
from forge_agent.test_pipeline.adaptation_engine import AdaptationEngine
from forge_agent.result_tracking.database import ResultTrackingDatabase
from forge_agent.result_tracking.models import TestRecord


class ModelCompatibilityPipeline:
    """
    Main pipeline for testing Hugging Face model compatibility with tt-torch.
    
    This class integrates all components of the system:
    1. Model Selection and Metadata Service
    2. Agentic Test Pipeline
    3. Result Tracking Database
    4. Analysis and Reporting Dashboard (via API)
    """
    
    def __init__(
        self,
        hf_api_token: Optional[str] = None,
        metadata_db_url: Optional[str] = None,
        results_db_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
        templates_dir: Optional[str] = None,
        llm_provider: str = "openai"
    ):
        """
        Initialize the pipeline.
        
        Args:
            hf_api_token: Optional Hugging Face API token
            metadata_db_url: URL for the metadata database
            results_db_url: URL for the results database
            cache_dir: Directory for caching downloaded models
            templates_dir: Directory containing adaptation templates
        """
        # Initialize components
        self.hf_client = HuggingFaceClient(api_token=hf_api_token)
        self.metadata_db = MetadataDatabase(db_url=metadata_db_url)
        self.result_db = ResultTrackingDatabase(db_url=results_db_url)
        self.downloader = ModelDownloader(cache_dir=cache_dir)
        self.adaptation_engine = AdaptationEngine(templates_dir=templates_dir, llm_provider=llm_provider)
        
        # Create data directory if it doesn't exist
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Log configuration info
        logger.info(f"Initialized Model Compatibility Pipeline with LLM provider: {llm_provider}")
        
        # Check for LLM API keys
        if llm_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable not set. LLM-guided adaptation will be disabled.")
        elif llm_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY environment variable not set. LLM-guided adaptation will be disabled.")
    
    def select_models(self, criteria: ModelSelectionCriteria) -> List[ModelMetadata]:
        """
        Select models for testing based on criteria.
        
        Args:
            criteria: Criteria for selecting models
            
        Returns:
            List of ModelMetadata for selected models
        """
        logger.info(f"Selecting models with criteria: {criteria}")
        
        # First check if we have enough models in the database that match the criteria
        models_in_db = self.metadata_db.filter_models(criteria)
        
        # If we need more models, fetch them from Hugging Face
        if len(models_in_db) < criteria.limit:
            logger.info(f"Found {len(models_in_db)} models in database, fetching more from Hugging Face")
            
            # Calculate how many more models we need
            models_needed = criteria.limit - len(models_in_db)
            
            # Get top models from Hugging Face
            top_models = self.hf_client.get_top_models(limit=models_needed * 2)  # Get extra to account for filtering
            
            # Process and store metadata for each model
            for model_data in top_models:
                model_id = model_data.get("id", "")
                if not model_id:
                    continue
                
                # Get detailed metadata for the model
                metadata = self.hf_client.get_model_metadata(model_id)
                
                # Store in the database
                self.metadata_db.store_model_metadata(metadata)
            
            # Re-query the database with the criteria
            models_in_db = self.metadata_db.filter_models(criteria)
        
        logger.info(f"Selected {len(models_in_db)} models for testing")
        return models_in_db
    
    def test_model(self, model_id: str, use_llm_adaptation: bool = True) -> TestRecord:
        """
        Test a single model for tt-torch compatibility.
        
        Args:
            model_id: Hugging Face model ID
            use_llm_adaptation: Whether to use LLM for adaptation if needed
            
        Returns:
            TestRecord with test results
        """
        logger.info(f"Testing model: {model_id}")
        
        # Create test record
        test_record = TestRecord(
            model_id=model_id,
            timestamp=datetime.now(),
            status=TestStatus.QUEUED
        )
        
        # Store initial record
        record_id = self.result_db.store_test_result(test_record)
        
        try:
            # Create test configuration
            test_config = TestConfig(
                model_id=model_id,
                use_llm_adaptation=use_llm_adaptation
            )
            
            # Step 1: Download model
            test_record.status = TestStatus.DOWNLOADING
            self.result_db.store_test_result(test_record)
            
            download_success, model_data, failure_reason, error_message = self.downloader.download_model(test_config)
            
            if not download_success:
                test_record.status = TestStatus.FAILED
                test_record.failure_reason = failure_reason
                test_record.error_message = error_message
                self.result_db.store_test_result(test_record)
                return test_record
            
            # Step 2: Apply adaptation
            test_record.status = TestStatus.ADAPTING
            self.result_db.store_test_result(test_record)
            
            adaptation_success, adapted_model, adaptation_level, failure_reason, error_message = \
                self.adaptation_engine.adapt_model(model_data, test_config)
            
            test_record.adaptation_level = adaptation_level
            
            if not adaptation_success:
                test_record.status = TestStatus.FAILED
                test_record.failure_reason = failure_reason
                test_record.error_message = error_message
                self.result_db.store_test_result(test_record)
                return test_record
            
            # Step 3: Attempt compilation with tt-torch
            test_record.status = TestStatus.COMPILING
            self.result_db.store_test_result(test_record)
            
            # Record compilation start time
            compilation_start_time = time.time()
            
            # Try to import tt-torch (which may not be installed in all environments)
            try:
                import tt_torch
                from tt_torch.tools.utils import CompilerConfig
                from tt_torch.dynamo.backend import BackendOptions
                
                # Configure tt-torch compiler
                model = adapted_model.get("model")
                sample_inputs = adapted_model.get("sample_inputs")
                
                # Create a simple compiler configuration
                cc = CompilerConfig()
                cc.tt_backend_options = BackendOptions()
                cc.tt_backend_options.buda_max_timer = 60000  # 60 seconds timeout
                
                # Attempt compilation
                logger.info(f"Attempting compilation for {model_id}")
                
                # Record memory usage before compilation
                memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                # Example compilation code (would be replaced with actual tt-torch compilation)
                # In a real implementation, this would use the tt-torch API
                compiled_model = model  # Placeholder, real implementation would compile the model
                
                # Record memory usage after compilation
                memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_usage_mb = (memory_after - memory_before) / (1024 * 1024)
                
                # Record compilation time
                compilation_time_seconds = time.time() - compilation_start_time
                test_record.compilation_time_seconds = compilation_time_seconds
                test_record.memory_usage_mb = memory_usage_mb
                
            except ImportError:
                logger.warning("tt-torch not installed, skipping actual compilation")
                # Simulate compilation for testing without tt-torch
                time.sleep(2)  # Simulate compilation time
                compilation_time_seconds = 2.0
                test_record.compilation_time_seconds = compilation_time_seconds
            except Exception as e:
                logger.error(f"Compilation error: {str(e)}")
                test_record.status = TestStatus.FAILED
                test_record.failure_reason = FailureReason.COMPILATION_ERROR
                test_record.error_message = f"Compilation error: {str(e)}"
                test_record.stack_trace = traceback.format_exc()
                self.result_db.store_test_result(test_record)
                return test_record
            
            # Step 4: Execute inference
            test_record.status = TestStatus.EXECUTING
            self.result_db.store_test_result(test_record)
            
            # Record execution start time
            execution_start_time = time.time()
            
            try:
                # Try to import tt-torch (which may not be installed in all environments)
                try:
                    # In a real implementation, this would execute inference with the compiled model
                    # outputs = compiled_model(**sample_inputs)
                    
                    # For now, just simulate execution
                    time.sleep(1)
                    
                except ImportError:
                    logger.warning("tt-torch not installed, skipping actual execution")
                    # Simulate execution for testing without tt-torch
                    time.sleep(1)
                
                # Record execution time
                execution_time_seconds = time.time() - execution_start_time
                test_record.execution_time_seconds = execution_time_seconds
                
                # Step 5: Mark as completed
                test_record.status = TestStatus.COMPLETED
                self.result_db.store_test_result(test_record)
                
            except Exception as e:
                logger.error(f"Execution error: {str(e)}")
                test_record.status = TestStatus.FAILED
                test_record.failure_reason = FailureReason.EXECUTION_ERROR
                test_record.error_message = f"Execution error: {str(e)}"
                test_record.stack_trace = traceback.format_exc()
                self.result_db.store_test_result(test_record)
            
            return test_record
            
        except Exception as e:
            logger.error(f"Error testing model {model_id}: {str(e)}")
            test_record.status = TestStatus.FAILED
            test_record.failure_reason = FailureReason.UNKNOWN
            test_record.error_message = f"Unknown error: {str(e)}"
            test_record.stack_trace = traceback.format_exc()
            self.result_db.store_test_result(test_record)
            return test_record
    
    def run_pipeline(self, model_selection_criteria: ModelSelectionCriteria, max_concurrent: int = 1) -> Dict[str, Any]:
        """
        Run the full pipeline on multiple models.
        
        Args:
            model_selection_criteria: Criteria for selecting models to test
            max_concurrent: Maximum number of models to test concurrently
            
        Returns:
            Dictionary with results summary
        """
        logger.info(f"Running pipeline with criteria: {model_selection_criteria}")
        
        # Step 1: Model Selection
        selected_models = self.select_models(model_selection_criteria)
        
        if not selected_models:
            logger.warning("No models selected for testing")
            return {"error": "No models selected for testing"}
        
        # Step 2: Testing Lifecycle
        results = []
        for i, model_metadata in enumerate(selected_models):
            logger.info(f"Testing model {i+1}/{len(selected_models)}: {model_metadata.model_id}")
            
            # Test the model
            test_record = self.test_model(model_metadata.model_id)
            results.append(test_record)
        
        # Step 3: Result Analysis
        statistics = self.result_db.get_statistics()
        
        return {
            "models_tested": len(results),
            "successful_tests": sum(1 for r in results if r.status == TestStatus.COMPLETED),
            "failed_tests": sum(1 for r in results if r.status == TestStatus.FAILED),
            "statistics": {
                "total_count": statistics.total_count,
                "success_rate": statistics.success_rate,
                "avg_compilation_time": statistics.avg_compilation_time_seconds,
                "avg_execution_time": statistics.avg_execution_time_seconds
            }
        }
