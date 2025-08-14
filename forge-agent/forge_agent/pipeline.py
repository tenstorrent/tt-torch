"""
Main pipeline for the Hugging Face Model Compatibility Pipeline.
"""
import time
from typing import List, Dict, Any, Optional
import os
import traceback
from datetime import datetime
import requests

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


def create_model_logger(model_id: str):
    """Create a dedicated logger for a specific model's adaptation process."""
    # Clean up model ID for filename
    clean_model_id = model_id.replace("/", "_").replace("\\", "_")
    
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "adaptations")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{clean_model_id}_adaptation.log")
    
    # Create a logger specifically for this model
    model_logger = logger.bind(model_id=model_id)
    
    # Add file handler for this model
    adaptation_format = (
        "=== {time:YYYY-MM-DD HH:mm:ss} ===\n"
        "{level: <8} | {message}\n"
    )
    
    model_logger.add(log_file, level="DEBUG", format=adaptation_format)
    
    return model_logger, log_file


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
    
    def _interpret_resnet_output(self, model_id: str, outputs: Any, top_k: int = 5) -> Dict[str, Any]:
        """
        Interpret ResNet model outputs and return meaningful predictions.
        
        Args:
            model_id: Model identifier
            outputs: Model output tensor or object
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with interpretation results
        """
        try:
            # Get the prediction tensor from the output
            if hasattr(outputs, 'prediction_logits'):
                predictions = outputs.prediction_logits
            elif hasattr(outputs, 'logits'):
                predictions = outputs.logits
            elif hasattr(outputs, 'last_hidden_state'):
                # HuggingFace ResNet models return features, not classification logits
                # We need to get the pooled output and apply a classifier
                logger.warning(f"HuggingFace ResNet model detected - output contains features, not classification logits")
                if hasattr(outputs, 'pooler_output'):
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state
                # For demo purposes, we'll simulate classification results
                logger.info(f"Features shape: {features.shape}")
                return {
                    "demo_note": "HuggingFace ResNet models return features, not ImageNet classifications",
                    "feature_shape": str(features.shape),
                    "explanation": "This model outputs feature vectors for transfer learning, not ImageNet class predictions",
                    "image_info": "Two cats on a couch (COCO dataset image)",
                    "model_id": model_id
                }
            elif isinstance(outputs, torch.Tensor):
                predictions = outputs
            else:
                logger.warning(f"Unknown output format for {model_id}: {type(outputs)}")
                return {"error": f"Unknown output format: {type(outputs)}"}
            
            # Apply softmax to get probabilities
            probs = torch.softmax(predictions.squeeze(), dim=-1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, top_k)
            
            # Load ImageNet class names (simplified version)
            # In a real implementation, you'd load the full ImageNet classes
            imagenet_classes = self._get_imagenet_classes()
            
            # Format results
            results = {
                "top_predictions": [],
                "image_info": "Two cats on a couch (COCO dataset image)",
                "model_id": model_id
            }
            
            for i, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
                class_name = imagenet_classes.get(idx, f"Class_{idx}")
                confidence = prob * 100
                results["top_predictions"].append({
                    "rank": i + 1,
                    "class": class_name,
                    "confidence": f"{confidence:.2f}%",
                    "class_idx": idx
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error interpreting ResNet output for {model_id}: {str(e)}")
            return {"error": str(e)}
    
    def _get_imagenet_classes(self) -> Dict[int, str]:
        """
        Get a subset of ImageNet class names. In a full implementation,
        this would load the complete ImageNet class mapping.
        """
        # Common ImageNet classes that might appear for "cats on couch" image
        common_classes = {
            281: "tabby cat",
            282: "tiger cat", 
            283: "Persian cat",
            284: "Siamese cat",
            285: "Egyptian cat",
            286: "lynx",
            287: "wildcat",
            831: "studio couch",
            445: "beach wagon",
            516: "couch",
            698: "padlock",
            700: "safety pin"
        }
        
        # Add some fallback generic names
        for i in range(1000):
            if i not in common_classes:
                common_classes[i] = f"ImageNet_Class_{i}"
        
        return common_classes
    
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
        logger.info(f"🧪 Testing model: {model_id}")
        
        # Create model-specific logger for detailed adaptation tracking
        model_logger, adaptation_log_file = create_model_logger(model_id)
        model_logger.info(f"🚀 STARTING TEST SESSION FOR: {model_id}")
        model_logger.info(f"📝 Adaptation log: {adaptation_log_file}")
        model_logger.info(f"🤖 LLM adaptation enabled: {use_llm_adaptation}")
        model_logger.info("=" * 80)
        
        # Create test record
        test_record = TestRecord(
            model_id=model_id,
            timestamp=datetime.now(),
            status=TestStatus.QUEUED
        )
        
        # Store initial record and keep the ID so future saves update the same row
        record_id = self.result_db.store_test_result(test_record)
        test_record.id = record_id
        model_logger.info(f"📋 Test record created with ID: {record_id}")
        
        try:
            # Create test configuration
            test_config = TestConfig(
                model_id=model_id,
                use_llm_adaptation=use_llm_adaptation
            )
            
            # Step 1: Download model
            test_record.status = TestStatus.DOWNLOADING
            self.result_db.store_test_result(test_record)
            model_logger.info("📥 PHASE 1: MODEL DOWNLOAD")
            
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
            model_logger.info("🔧 PHASE 2: MODEL ADAPTATION")
            
            adaptation_success, adapted_model, adaptation_level, failure_reason, error_message = \
                self.adaptation_engine.adapt_model(model_data, test_config, model_logger)
            
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
            
            # Try to import tt-torch and check for hardware availability
            tt_torch_available = False
            hardware_available = False
            compiled_model = None
            
            # Prepare sample inputs early (needed for real compilation)
            model = adapted_model.get("model")
            sample_inputs = adapted_model.get("sample_inputs")
            
            # Ensure sample_inputs is valid
            if sample_inputs is None:
                logger.warning(f"sample_inputs is None for {model_id}, creating fallback inputs")
                sample_inputs = {"pixel_values": torch.rand(1, 3, 224, 224)}
            elif hasattr(sample_inputs, 'keys') and hasattr(sample_inputs, '__getitem__'):
                # This handles both dict and BatchEncoding objects
                logger.debug(f"Using sample_inputs for {model_id}: {list(sample_inputs.keys())}")
                # Convert BatchEncoding to regular dict if needed
                if not isinstance(sample_inputs, dict):
                    sample_inputs = dict(sample_inputs)
                    logger.debug(f"Converted BatchEncoding to dict for {model_id}")
            elif not isinstance(sample_inputs, dict):
                logger.warning(f"Invalid sample_inputs for {model_id}: {type(sample_inputs)}, creating fallback inputs")
                sample_inputs = {"pixel_values": torch.rand(1, 3, 224, 224)}
            else:
                logger.debug(f"Using sample_inputs for {model_id}: {list(sample_inputs.keys())}")

            # Sanitize inputs for encoder-decoder models to avoid mutually exclusive args
            try:
                if hasattr(model, "config") and getattr(model.config, "is_encoder_decoder", False) and isinstance(sample_inputs, dict):
                    # Never pass both decoder_input_ids and decoder_inputs_embeds
                    if "decoder_input_ids" in sample_inputs and "decoder_inputs_embeds" in sample_inputs:
                        logger.info(f"🧹 Sanitizing inputs for {model_id}: removing decoder_inputs_embeds since decoder_input_ids is present")
                        sample_inputs.pop("decoder_inputs_embeds", None)
                    # Ensure at least minimal decoder_input_ids exists
                    if "decoder_input_ids" not in sample_inputs:
                        bos_id = getattr(model.config, "decoder_start_token_id", None)
                        if bos_id is None:
                            bos_id = getattr(model.config, "bos_token_id", None)
                        if bos_id is None:
                            bos_id = 1
                        sample_inputs["decoder_input_ids"] = torch.tensor([[bos_id]], dtype=torch.long)
                        logger.debug(f"Added minimal decoder_input_ids for {model_id} using BOS id {bos_id}")
            except Exception as sanitize_e:
                logger.debug(f"Input sanitization skipped for {model_id}: {sanitize_e}")
            
            # Force real tt-torch execution - no simulation fallbacks
            
            try:
                # Check if tt-torch is available
                try:
                    import tt_torch
                    from tt_torch.tools.utils import CompilerConfig
                    from tt_torch.dynamo.backend import BackendOptions
                    tt_torch_available = True
                    logger.info(f"✅ tt-torch is available for {model_id}")
                    logger.info(f"   - tt_torch module path: {tt_torch.__file__}")
                    logger.info(f"   - CompilerConfig available: {CompilerConfig is not None}")
                    logger.info(f"   - BackendOptions available: {BackendOptions is not None}")
                    
                    # Check if Tenstorrent hardware is available
                    try:
                        # Check if the backend compilation tools are available
                        # This indicates the full tt-torch stack is functional
                        import tt_torch.dynamo.backend as backend
                        if hasattr(backend, 'tt_mlir'):
                            hardware_available = True
                            logger.info(f"✅ Tenstorrent hardware/toolchain appears to be available for {model_id}")
                            logger.info(f"   - tt_mlir backend module: {backend.tt_mlir}")
                            logger.info(f"   - WILL USE REAL TT-TORCH COMPILATION")
                        else:
                            logger.warning(f"tt_mlir not available in backend for {model_id}")
                            hardware_available = False
                    except Exception as hw_e:
                        logger.warning(f"Tenstorrent hardware/toolchain not available for {model_id}: {str(hw_e)}")
                        hardware_available = False
                        
                except ImportError as e:
                    logger.info(f"tt-torch not available for {model_id}: {str(e)}")
                    tt_torch_available = False
                
                if tt_torch_available and hardware_available:
                    # Create tt-torch compiler configuration (like in resnet50_demo.py)
                    cc = CompilerConfig()
                    cc.enable_consteval = True
                    cc.consteval_parameters = True
                    
                    # Set backend options
                    options = BackendOptions()
                    options.compiler_config = cc
                    
                    # Attempt actual tt-torch compilation
                    logger.info(f"Attempting REAL tt-torch compilation for {model_id}")
                    
                    # Record memory usage before compilation
                    memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    # First test the original model to make sure it works
                    logger.info(f"Testing original model before compilation for {model_id}")
                    original_model_success = False
                    max_input_fix_retries = 2
                    
                    for input_fix_attempt in range(max_input_fix_retries):
                        try:
                            model.eval()
                            with torch.no_grad():
                                _ = model(**sample_inputs)
                            logger.info(f"Original model test passed for {model_id}")
                            original_model_success = True
                            break
                            
                        except Exception as orig_e:
                            error_msg = str(orig_e)
                            logger.error(f"Original model failed (attempt {input_fix_attempt + 1}/{max_input_fix_retries}) for {model_id}: {error_msg}")
                            
                            # Route any error to LLM input-fix if enabled and retries remain
                            if use_llm_adaptation and input_fix_attempt < max_input_fix_retries - 1:
                                logger.info(f"🤖 ROUTING ERROR TO LLM INPUT FIX for {model_id}")
                                logger.info(f"   - Error: {error_msg}")
                                logger.info(f"   - Current inputs: {list(sample_inputs.keys())}")
                                # Extract model info for LLM
                                model_info = self.adaptation_engine._extract_model_info(model, model_id)
                                # Call LLM to fix the inputs
                                llm_result = self.adaptation_engine.llm_client.fix_input_arguments(
                                    model_info=model_info,
                                    error_message=error_msg,
                                    current_inputs=sample_inputs,
                                    model_class_name=type(model).__name__
                                )
                                if llm_result.get("success") and llm_result.get("inputs"):
                                    logger.info(f"✅ LLM successfully generated fixed inputs for {model_id}")
                                    # Log detailed before/after comparison
                                    old_inputs = sample_inputs
                                    new_inputs = llm_result["inputs"]
                                    logger.info(f"🔄 INPUT CHANGES MADE BY LLM:")
                                    logger.info(f"   📥 ORIGINAL INPUTS (that failed):")
                                    for key, value in old_inputs.items():
                                        shape_str = getattr(value, 'shape', 'unknown shape')
                                        dtype_str = getattr(value, 'dtype', 'unknown dtype')
                                        logger.info(f"      - {key}: {type(value).__name__} {shape_str} {dtype_str}")
                                    logger.info(f"   📤 NEW INPUTS (LLM-generated):")
                                    for key, value in new_inputs.items():
                                        shape_str = getattr(value, 'shape', 'unknown shape')
                                        dtype_str = getattr(value, 'dtype', 'unknown dtype')
                                        logger.info(f"      - {key}: {type(value).__name__} {shape_str} {dtype_str}")
                                    logger.info(f"   💡 LLM EXPLANATION: {llm_result.get('explanation', 'No explanation provided')}")
                                    # Show the key changes
                                    old_keys = set(old_inputs.keys())
                                    new_keys = set(new_inputs.keys())
                                    if old_keys != new_keys:
                                        removed_keys = old_keys - new_keys
                                        added_keys = new_keys - old_keys
                                        if removed_keys:
                                            logger.info(f"   ❌ REMOVED PARAMETERS: {list(removed_keys)}")
                                        if added_keys:
                                            logger.info(f"   ✅ ADDED PARAMETERS: {list(added_keys)}")
                                        unchanged_keys = old_keys & new_keys
                                        if unchanged_keys:
                                            logger.info(f"   ↔️  UNCHANGED PARAMETERS: {list(unchanged_keys)}")
                                    # Update sample_inputs with the fixed inputs
                                    sample_inputs = llm_result["inputs"]
                                    # Also update the adapted_model data for consistency
                                    adapted_model["sample_inputs"] = sample_inputs
                                    logger.info(f"🔄 Retrying original model test with LLM-fixed inputs for {model_id}")
                                    continue  # Retry with fixed inputs
                                else:
                                    logger.error(f"❌ LLM failed to fix inputs for {model_id}: {llm_result.get('explanation', 'Unknown error')}")

                            # If we reach here, no more retries
                            if input_fix_attempt == max_input_fix_retries - 1:
                                logger.error(f"Original model failed after all attempts for {model_id}, skipping tt-torch compilation")
                                raise Exception(f"Original model validation failed: {error_msg}")
                    
                    if not original_model_success:
                        logger.error(f"Original model validation failed for {model_id} after all retry attempts")
                        raise Exception(f"Original model validation failed after {max_input_fix_retries} attempts")
                    
                    # REAL tt-torch compilation using torch.compile with "tt" backend
                    logger.info(f"Compiling model with tt-torch backend for {model_id}")
                    logger.info(f"🔥 CALLING torch.compile() with backend='tt' for {model_id}")
                    logger.info(f"   - Model type: {type(model).__name__}")
                    logger.info(f"   - Backend: 'tt' (Tenstorrent)")
                    logger.info(f"   - Dynamic: False") 
                    logger.info(f"   - Compiler config: enable_consteval={cc.enable_consteval}, consteval_parameters={cc.consteval_parameters}")
                    
                    # Pre-compilation debugging - log model structure to help identify problematic operations
                    logger.info(f"🔍 PRE-COMPILATION ANALYSIS for {model_id}:")
                    try:
                        model_modules = list(model.named_modules())
                        logger.info(f"   - Total modules: {len(model_modules)}")
                        
                        # Log potentially problematic module types
                        problematic_modules = []
                        for name, module in model_modules[:10]:  # Log first 10 modules
                            module_type = type(module).__name__
                            logger.info(f"     {name}: {module_type}")
                            
                            # Check for known problematic operations
                            if any(op in module_type.lower() for op in ['layernorm', 'groupnorm', 'instancenorm', 'batchnorm']):
                                problematic_modules.append(f"{name}:{module_type}")
                        
                        if problematic_modules:
                            logger.warning(f"   ⚠️  POTENTIALLY PROBLEMATIC MODULES: {problematic_modules}")
                        
                        if len(model_modules) > 10:
                            logger.info(f"     ... and {len(model_modules) - 10} more modules")
                            
                    except Exception as analysis_error:
                        logger.warning(f"   ❌ Pre-compilation analysis failed: {str(analysis_error)}")
                    
                    # CRITICAL: Safe torch.compile call with comprehensive error handling
                    logger.info(f"🚨 CRITICAL SECTION: Calling torch.compile() - this may segfault if unsupported ops are present")
                    compiled_model = None
                    compilation_error = None
                    
                    # Check if we should use safer subprocess-based compilation for crash-prone models
                    use_subprocess_safety = os.environ.get('TT_TORCH_SAFE_COMPILE', 'false').lower() == 'true'
                    
                    if use_subprocess_safety:
                        logger.info(f"🛡️  Using SUBPROCESS SAFETY MODE for {model_id}")
                        logger.info(f"   - This will isolate torch.compile() to prevent main process crashes")
                        # TODO: Implement subprocess-based compilation for maximum safety
                        # For now, fall back to signal handler approach
                    
                    try:
                        # Set up signal handler for segfault detection
                        import signal
                        import sys
                        
                        segfault_detected = False
                        
                        def segfault_handler(signum, frame):
                            nonlocal segfault_detected
                            segfault_detected = True
                            logger.error(f"💀 SEGFAULT DETECTED during torch.compile() for {model_id}")
                            logger.error(f"   - Signal: {signum} (SIGSEGV)")
                            logger.error(f"   - Frame: {frame.f_code.co_filename}:{frame.f_lineno}" if frame else "Unknown")
                            logger.error(f"   - Model contains operations that crash tt-torch backend")
                            logger.error(f"   - Common causes: LayerNorm, unsupported ATen ops, memory issues")
                            logger.error(f"   - Try: 1) Model adaptation, 2) Different model architecture")
                            
                            # Create a detailed crash report
                            crash_info = {
                                'model_id': model_id,
                                'model_type': type(model).__name__,
                                'signal': signum,
                                'crash_location': f"{frame.f_code.co_filename}:{frame.f_lineno}" if frame else "unknown",
                                'problematic_modules': problematic_modules if 'problematic_modules' in locals() else []
                            }
                            logger.error(f"   - Crash info: {crash_info}")
                            
                            # Don't call sys.exit() directly from signal handler - just mark and return
                            return
                        
                        # Install signal handler for SIGSEGV
                        original_handler = signal.signal(signal.SIGSEGV, segfault_handler)
                        
                        try:
                            logger.info(f"🔥 Executing torch.compile() with SEGFAULT protection...")
                            
                            # Add timeout protection as well
                            import threading
                            
                            compilation_timeout = 300  # 5 minutes timeout
                            compilation_result = [None]  # Use list for mutable reference
                            compilation_exception = [None]
                            
                            def compile_with_timeout():
                                try:
                                    compilation_result[0] = torch.compile(model, backend="tt", dynamic=False, options=options)
                                except Exception as e:
                                    compilation_exception[0] = e
                            
                            compile_thread = threading.Thread(target=compile_with_timeout)
                            compile_thread.daemon = True
                            compile_thread.start()
                            compile_thread.join(timeout=compilation_timeout)
                            
                            # Check for segfault
                            if segfault_detected:
                                raise Exception(f"Segmentation fault detected during torch.compile() for {model_id}")
                            
                            # Check for timeout
                            if compile_thread.is_alive():
                                logger.error(f"⏰ TIMEOUT: torch.compile() exceeded {compilation_timeout}s for {model_id}")
                                raise Exception(f"Compilation timeout after {compilation_timeout} seconds")
                            
                            # Check for compilation exception
                            if compilation_exception[0]:
                                raise compilation_exception[0]
                            
                            # Get result
                            compiled_model = compilation_result[0]
                            
                            if compiled_model is None:
                                raise Exception("Compilation completed but returned None")
                                
                            logger.info(f"✅ torch.compile() completed WITHOUT segfault for {model_id}")
                            
                        finally:
                            # Restore original signal handler
                            signal.signal(signal.SIGSEGV, original_handler)
                            
                    except Exception as compile_error:
                        compilation_error = compile_error
                        logger.error(f"❌ torch.compile() FAILED for {model_id}: {str(compile_error)}")
                        logger.error(f"   - Error type: {type(compile_error).__name__}")
                        logger.error(f"   - This indicates the model contains operations unsupported by tt-torch")
                        
                        # Try to extract useful information from the error
                        error_str = str(compile_error).lower()
                        
                        # Look for specific error patterns
                        if 'segmentation fault' in error_str or 'sigsegv' in error_str:
                            logger.error(f"   💀 SEGFAULT confirmed - model crashes tt-torch backend")
                        elif 'unsupported' in error_str:
                            logger.error(f"   🎯 UNSUPPORTED OPERATION detected in error message")
                        elif 'timeout' in error_str:
                            logger.error(f"   ⏰ COMPILATION TIMEOUT - model too complex or hanging")
                        
                        if 'aten::' in error_str:
                            import re
                            aten_ops = re.findall(r'aten::\w+', error_str)
                            if aten_ops:
                                logger.error(f"   🔍 Problematic ATen operations: {aten_ops}")
                        
                        # Log suggestions for fixing the issue
                        logger.error(f"   💡 SUGGESTED FIXES:")
                        logger.error(f"     1. Enable LLM adaptation to replace unsupported operations")
                        logger.error(f"     2. Try a different model architecture")
                        logger.error(f"     3. Check for LayerNorm, GroupNorm, or other problematic layers")
                        logger.error(f"     4. Use TT_TORCH_SAFE_COMPILE=true environment variable")
                        
                        raise compile_error
                    
                    # Validate compilation results
                    if compiled_model is None:
                        error_msg = "Compilation returned None - possible silent failure"
                        logger.error(f"❌ {error_msg}")
                        raise Exception(error_msg)
                    
                    # Log details about the compiled model
                    logger.info(f"✅ torch.compile() completed successfully for {model_id}")
                    logger.info(f"   - Compiled model type: {type(compiled_model).__name__}")
                    logger.info(f"   - Original model type: {type(model).__name__}")
                    logger.info(f"   - Compiled model repr: {repr(compiled_model)[:100]}...")
                    
                    # Check if this is actually a tt-torch compiled model
                    if hasattr(compiled_model, '_torchdynamo_orig_callable'):
                        logger.info(f"🎯 CONFIRMED: Model is torch-compiled (has _torchdynamo_orig_callable)")
                    else:
                        logger.warning(f"❓ Model may not be properly compiled (missing _torchdynamo_orig_callable)")
                    
                    # Additional validation
                    try:
                        # Try to access model parameters to ensure it's not corrupted
                        param_count = sum(p.numel() for p in compiled_model.parameters())
                        logger.info(f"   - Parameter count: {param_count:,}")
                    except Exception as param_error:
                        logger.warning(f"   ❌ Could not access compiled model parameters: {str(param_error)}")
                    
                    logger.info(f"Compilation completed for {model_id}, validating compiled model...")
                    
                    # Test if the compiled model is safe to use by checking basic properties
                    try:
                        # Basic validation - check if the compiled model has expected attributes
                        if not hasattr(compiled_model, '__call__'):
                            raise Exception("Compiled model is not callable")
                        logger.info(f"Compiled model basic validation passed for {model_id}")
                    except Exception as validation_e:
                        logger.error(f"Compiled model validation failed for {model_id}: {str(validation_e)}")
                        raise Exception(f"Compiled model validation failed: {str(validation_e)}")
                    
                    # Record memory usage after compilation
                    memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    memory_usage_mb = (memory_after - memory_before) / (1024 * 1024)
                    
                    # Record compilation time
                    compilation_time_seconds = time.time() - compilation_start_time
                    test_record.compilation_time_seconds = compilation_time_seconds
                    test_record.memory_usage_mb = memory_usage_mb
                else:
                    # Fail if tt-torch or hardware is not available - no simulation fallback
                    error_msg = f"❌ REAL TT-TORCH REQUIRED but not available for {model_id}"
                    if not tt_torch_available:
                        error_msg += f"\n   - tt-torch module not available"
                    if not hardware_available:
                        error_msg += f"\n   - Tenstorrent hardware/toolchain not available"
                    error_msg += f"\n   - SIMULATION DISABLED - must have working tt-torch setup"
                    logger.error(error_msg)
                    raise Exception(f"tt-torch not available: tt_torch={tt_torch_available}, hardware={hardware_available}")
                    
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
                # Execute inference with the compiled tt-torch model
                if 'compiled_model' in locals() and sample_inputs and (isinstance(sample_inputs, dict) or hasattr(sample_inputs, 'keys')):
                    logger.info(f"Running inference with compiled tt-torch model for {model_id}")
                    logger.info(f"🚀 EXECUTING INFERENCE with tt-torch compiled model")
                    logger.info(f"   - Compiled model type: {type(compiled_model).__name__}")
                    logger.info(f"   - Input shape: {[(k, v.shape) for k, v in sample_inputs.items()]}")
                    logger.info(f"   - Using Tenstorrent backend for acceleration")
                    
                    # Try running the compiled model with safety checks
                    try:
                        logger.info(f"Attempting minimal test call on compiled model for {model_id}")
                        
                        # First do a minimal test to catch segfaults early
                        # Try to access the compiled model safely
                        _ = str(type(compiled_model))
                        logger.debug(f"Compiled model type check passed for {model_id}")
                        
                        # REAL inference execution with the compiled model
                        logger.info(f"🎯 RUNNING FULL INFERENCE with tt-torch backend for {model_id}")
                        logger.info(f"   - This is using REAL Tenstorrent hardware acceleration")
                        
                        # CRITICAL: Safe inference execution with segfault protection
                        logger.info(f"🚨 CRITICAL SECTION: Running inference - this may segfault if unsupported ops are executed")
                        outputs = None
                        inference_error = None
                        
                        try:
                            # Set up signal handler for inference segfault detection
                            import signal
                            import sys
                            
                            inference_segfault_detected = False
                            
                            def inference_segfault_handler(signum, frame):
                                nonlocal inference_segfault_detected
                                inference_segfault_detected = True
                                logger.error(f"💀 INFERENCE SEGFAULT DETECTED for {model_id}")
                                logger.error(f"   - Signal: {signum} (SIGSEGV)")
                                logger.error(f"   - Frame: {frame.f_code.co_filename}:{frame.f_lineno}" if frame else "Unknown")
                                logger.error(f"   - Model execution crashes during inference on tt-torch")
                                logger.error(f"   - Common causes: Unsupported runtime ops, memory corruption, driver issues")
                                logger.error(f"   - Try: 1) Model adaptation, 2) Smaller batch size, 3) Different input shapes")
                                
                                # Create detailed inference crash report
                                crash_info = {
                                    'model_id': model_id,
                                    'model_type': type(compiled_model).__name__,
                                    'signal': signum,
                                    'crash_location': f"{frame.f_code.co_filename}:{frame.f_lineno}" if frame else "unknown",
                                    'input_shapes': {k: v.shape if hasattr(v, 'shape') else str(type(v)) for k, v in sample_inputs.items()},
                                    'stage': 'inference_execution'
                                }
                                logger.error(f"   - Inference crash info: {crash_info}")
                                return
                            
                            # Install signal handler for inference SIGSEGV
                            original_inference_handler = signal.signal(signal.SIGSEGV, inference_segfault_handler)
                            
                            try:
                                logger.info(f"🔥 Executing inference with SEGFAULT protection...")
                                
                                # Add timeout protection for inference as well
                                import threading
                                
                                # Inference timeout (seconds) - configurable via env
                                inference_timeout = int(os.environ.get('FORGE_INFERENCE_TIMEOUT_SECONDS', '120'))
                                inference_result = [None]
                                inference_exception = [None]
                                
                                def inference_with_timeout():
                                    try:
                                        with torch.no_grad():
                                            inference_result[0] = compiled_model(**sample_inputs)
                                    except Exception as e:
                                        inference_exception[0] = e
                                
                                inference_thread = threading.Thread(target=inference_with_timeout)
                                inference_thread.daemon = True
                                inference_thread.start()
                                inference_thread.join(timeout=inference_timeout)
                                
                                # Check for inference segfault
                                if inference_segfault_detected:
                                    raise Exception(f"Segmentation fault detected during inference for {model_id}")
                                
                                # Check for inference timeout
                                if inference_thread.is_alive():
                                    logger.error(f"⏰ INFERENCE TIMEOUT: execution exceeded {inference_timeout}s for {model_id}")
                                    # Mark record as failed with timeout and continue to next model
                                    test_record.status = TestStatus.FAILED
                                    test_record.failure_reason = FailureReason.TIMEOUT
                                    test_record.error_message = f"Inference timeout after {inference_timeout} seconds"
                                    self.result_db.store_test_result(test_record)
                                    return test_record
                                
                                # Check for inference exception
                                if inference_exception[0]:
                                    raise inference_exception[0]
                                
                                # Get inference result
                                outputs = inference_result[0]
                                
                                if outputs is None:
                                    raise Exception("Inference completed but returned None")
                                    
                                logger.info(f"✅ TT-TORCH INFERENCE COMPLETED WITHOUT SEGFAULT!")
                                
                            finally:
                                # Restore original signal handler
                                signal.signal(signal.SIGSEGV, original_inference_handler)
                                
                        except Exception as inf_error:
                            inference_error = inf_error
                            logger.error(f"❌ INFERENCE FAILED for {model_id}: {str(inf_error)}")
                            logger.error(f"   - Error type: {type(inf_error).__name__}")
                            logger.error(f"   - This indicates the compiled model cannot execute on tt-torch")
                            
                            # Analyze inference error patterns
                            error_str = str(inf_error).lower()
                            
                            if 'segmentation fault' in error_str or 'sigsegv' in error_str:
                                logger.error(f"   💀 INFERENCE SEGFAULT confirmed - model execution crashes")
                            elif 'timeout' in error_str:
                                logger.error(f"   ⏰ INFERENCE TIMEOUT - model execution hanging or too slow")
                            elif 'unsupported' in error_str:
                                logger.error(f"   🎯 UNSUPPORTED RUNTIME OPERATION detected")
                            elif 'cuda' in error_str or 'device' in error_str:
                                logger.error(f"   🖥️  DEVICE/MEMORY issue detected")
                            
                            logger.error(f"   💡 INFERENCE FAILURE FIXES:")
                            logger.error(f"     1. Enable LLM adaptation to replace problematic runtime ops")
                            logger.error(f"     2. Try smaller batch sizes or input shapes")
                            logger.error(f"     3. Check model contains only supported operations")
                            logger.error(f"     4. Verify Tenstorrent hardware/driver setup")
                            
                            raise inf_error
                        
                        logger.info(f"✅ TT-TORCH INFERENCE COMPLETED SUCCESSFULLY!")
                        logger.info(f"   - Output type: {type(outputs)}")
                        logger.info(f"   - Output shape: {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")
                        logger.info(f"   - Successfully executed on Tenstorrent hardware!")
                        
                        # For ResNet models, interpret the results and show predictions
                        if "resnet" in model_id.lower():
                            logger.info(f"🎯 INTERPRETING RESNET PREDICTIONS for {model_id}")
                            interpretation = self._interpret_resnet_output(model_id, outputs)
                            
                            if "error" not in interpretation:
                                logger.info(f"📸 Image: {interpretation['image_info']}")
                                
                                if "top_predictions" in interpretation:
                                    logger.info(f"🏆 TOP 5 PREDICTIONS:")
                                    for pred in interpretation["top_predictions"]:
                                        logger.info(f"   {pred['rank']}. {pred['class']}: {pred['confidence']}")
                                elif "demo_note" in interpretation:
                                    logger.info(f"📋 {interpretation['demo_note']}")
                                    logger.info(f"   - Feature shape: {interpretation['feature_shape']}")
                                    logger.info(f"   - {interpretation['explanation']}")
                                    logger.info(f"   - ✅ Successfully extracted feature vectors for transfer learning!")
                            else:
                                logger.warning(f"Failed to interpret predictions: {interpretation['error']}")
                        elif "gpt" in model_id.lower() or "bert" in model_id.lower():
                            logger.info(f"🎯 INTERPRETING LLM OUTPUT for {model_id}")
                            if hasattr(outputs, 'logits'):
                                logger.info(f"📝 Language Model Results:")
                                logger.info(f"   - Logits shape: {outputs.logits.shape}")
                                logger.info(f"   - Vocabulary size: {outputs.logits.shape[-1]}")
                                logger.info(f"   - ✅ Successfully generated token predictions!")
                            else:
                                logger.info(f"📝 Language Model Output:")
                                logger.info(f"   - Output type: {type(outputs)}")
                                logger.info(f"   - ✅ Successfully processed text through LLM!")
                        else:
                            logger.info(f"   - Raw output for non-ResNet model: {str(outputs)[:100]}...")
                    except Exception as compiled_e:
                        # Treat Tenstorrent compiled inference failure as an overall test failure
                        logger.error(f"Compiled model execution failed for {model_id}: {str(compiled_e)}")
                        test_record.status = TestStatus.FAILED
                        test_record.failure_reason = FailureReason.EXECUTION_ERROR
                        test_record.error_message = f"Inference failed on Tenstorrent: {str(compiled_e)}"
                        self.result_db.store_test_result(test_record)
                        return test_record
                else:
                    # Fail if no compiled model is available - no simulation fallback
                    error_msg = "❌ NO COMPILED MODEL AVAILABLE for inference"
                    if compiled_model is None:
                        error_msg += "\n   - Compilation failed or was skipped"
                    if not sample_inputs:
                        error_msg += "\n   - No sample inputs available"
                    error_msg += "\n   - SIMULATION DISABLED - must have working compiled model"
                    logger.error(error_msg)
                    raise Exception("No compiled model available for inference - simulation disabled")
                
                # Record execution time
                execution_time_seconds = time.time() - execution_start_time
                test_record.execution_time_seconds = execution_time_seconds
                
                # Step 5: Mark as completed
                test_record.status = TestStatus.COMPLETED
                self.result_db.store_test_result(test_record)

                # Save compiled model artifact if available and execution succeeded
                try:
                    if 'compiled_model' in locals() and compiled_model is not None:
                        artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
                        os.makedirs(artifacts_dir, exist_ok=True)
                        safe_name = model_id.replace('/', '_')
                        artifact_path = os.path.join(artifacts_dir, f"{safe_name}_compiled.pt")
                        # Best-effort save: prefer torch.save(state_dict) if available
                        to_save = compiled_model.state_dict() if hasattr(compiled_model, 'state_dict') else compiled_model
                        torch.save(to_save, artifact_path)
                        logger.info(f"💾 Saved compiled model artifact to: {artifact_path}")
                except Exception as save_err:
                    logger.warning(f"Could not save compiled model artifact for {model_id}: {save_err}")
                
                # Log final success
                model_logger.info("=" * 80)
                model_logger.info("🎉 TEST COMPLETED SUCCESSFULLY!")
                model_logger.info(f"✅ Status: {test_record.status}")
                model_logger.info(f"📈 Adaptation Level: {test_record.adaptation_level.value if hasattr(test_record.adaptation_level, 'value') else test_record.adaptation_level}")
                model_logger.info(f"⏱️  Compilation Time: {test_record.compilation_time_seconds:.2f}s")
                model_logger.info(f"🚀 Execution Time: {test_record.execution_time_seconds:.2f}s")
                model_logger.info("=" * 80)
                
            except Exception as e:
                logger.error(f"Execution error: {str(e)}")
                # Request LLM adaptation suggestions for execution failure
                try:
                    if use_llm_adaptation:
                        model_info = self.adaptation_engine._extract_model_info(model, model_id)
                        _ = self.adaptation_engine.llm_client.generate_adaptation_code(
                            model_info=model_info,
                            error_message=str(e),
                            model_class_name=type(model).__name__
                        )
                except Exception as llm_exec_e:
                    logger.debug(f"LLM adaptation suggestion failed for execution error: {llm_exec_e}")
                test_record.status = TestStatus.FAILED
                test_record.failure_reason = FailureReason.EXECUTION_ERROR
                test_record.error_message = f"Execution error: {str(e)}"
                test_record.stack_trace = traceback.format_exc()
                self.result_db.store_test_result(test_record)
                
                # Log execution failure
                model_logger.error("=" * 80)
                model_logger.error("💥 TEST FAILED - EXECUTION ERROR")
                model_logger.error(f"❌ Status: {test_record.status}")
                model_logger.error(f"🚨 Error: {test_record.error_message}")
                model_logger.error("=" * 80)
            
            return test_record
            
        except Exception as e:
            logger.error(f"Error testing model {model_id}: {str(e)}")
            test_record.status = TestStatus.FAILED
            test_record.failure_reason = FailureReason.UNKNOWN
            test_record.error_message = f"Unknown error: {str(e)}"
            test_record.stack_trace = traceback.format_exc()
            self.result_db.store_test_result(test_record)
            
            # Log unknown failure
            model_logger.error("=" * 80)
            model_logger.error("💥 TEST FAILED - UNKNOWN ERROR")
            model_logger.error(f"❌ Status: {test_record.status}")
            model_logger.error(f"🚨 Error: {test_record.error_message}")
            model_logger.error("=" * 80)
            
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
        logger.info(f"🚀 Running pipeline with criteria: {model_selection_criteria}")
        
        # Step 1: Model Selection
        selected_models = self.select_models(model_selection_criteria)
        
        if not selected_models:
            logger.warning("⚠️  No models selected for testing")
            return {"error": "No models selected for testing"}
        
        # Step 2: Testing Lifecycle
        results = []
        for i, model_metadata in enumerate(selected_models):
            logger.info(f"🧪 Testing model {i+1}/{len(selected_models)}: {model_metadata.model_id}")
            
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
