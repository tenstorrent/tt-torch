"""
Module for downloading models from Hugging Face.
"""
import os
from typing import Dict, Optional, Tuple, Any

from huggingface_hub import snapshot_download, hf_hub_download
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
from loguru import logger

from forge_agent.test_pipeline.models import TestConfig, FailureReason, TestStatus


class ModelDownloader:
    """Downloads models from Hugging Face and prepares them for testing."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the model downloader.
        
        Args:
            cache_dir: Directory to use for caching downloaded models
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/forge-agent")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def download_model(self, test_config: TestConfig) -> Tuple[bool, Optional[Dict[str, Any]], Optional[FailureReason], Optional[str]]:
        """
        Download a model from Hugging Face.
        
        Args:
            test_config: Configuration for the model test
            
        Returns:
            Tuple containing:
            - Success flag (True if successful)
            - Dictionary with model objects (config, model, tokenizer) if successful
            - Failure reason enum if failed
            - Error message if failed
        """
        model_id = test_config.model_id
        logger.info(f"Downloading model: {model_id}")
        
        try:
            # Download the model files
            snapshot_path = snapshot_download(
                repo_id=model_id,
                cache_dir=self.cache_dir,
                local_dir=os.path.join(self.cache_dir, model_id.replace("/", "_")),
                local_dir_use_symlinks=False
            )
            
            logger.info(f"Model downloaded to: {snapshot_path}")
            
            # Try to load the model configuration
            config = AutoConfig.from_pretrained(snapshot_path)
            
            # Attempt to load the tokenizer if it's a text model
            tokenizer = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
            except Exception as e:
                logger.warning(f"Could not load tokenizer for {model_id}: {str(e)}")
            
            # Attempt to load the model
            model = None
            try:
                # Try different model classes based on common architectures
                model_classes = [
                    AutoModel, 
                    AutoModelForCausalLM, 
                    AutoModelForSequenceClassification
                ]
                
                for model_class in model_classes:
                    try:
                        model = model_class.from_pretrained(snapshot_path)
                        break
                    except Exception as e:
                        logger.debug(f"Failed to load with {model_class.__name__}: {str(e)}")
                
                if model is None:
                    # If all attempts failed, try one more time with the generic loader
                    model = AutoModel.from_pretrained(snapshot_path)
            
            except Exception as e:
                logger.error(f"All attempts to load model {model_id} failed: {str(e)}")
                return False, None, FailureReason.DOWNLOAD_ERROR, f"Failed to load model: {str(e)}"
            
            # Create sample inputs if needed
            sample_inputs = self._create_sample_inputs(model, tokenizer, test_config)
            
            return True, {
                "config": config,
                "model": model,
                "tokenizer": tokenizer,
                "sample_inputs": sample_inputs,
                "path": snapshot_path
            }, None, None
            
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {str(e)}")
            return False, None, FailureReason.DOWNLOAD_ERROR, f"Download failed: {str(e)}"
    
    def _create_sample_inputs(self, model: Any, tokenizer: Any, test_config: TestConfig) -> Dict[str, torch.Tensor]:
        """
        Create sample inputs for model testing.
        
        Args:
            model: The loaded model
            tokenizer: The loaded tokenizer
            test_config: Configuration for the model test
            
        Returns:
            Dictionary of sample inputs
        """
        sample_inputs = {}
        
        # Use provided shapes if available
        if test_config.sample_input_shape:
            for name, shape in test_config.sample_input_shape.items():
                sample_inputs[name] = torch.rand(*shape)
            return sample_inputs
        
        # Otherwise, try to infer appropriate inputs based on model type
        try:
            if tokenizer is not None:
                # For text models, create a simple input
                if hasattr(tokenizer, "encode") or hasattr(tokenizer, "encode_plus"):
                    text = "This is a sample input for testing the model."
                    inputs = tokenizer(text, return_tensors="pt")
                    sample_inputs = inputs
                    return sample_inputs
            
            # For other model types, try to infer from config
            config = model.config if hasattr(model, "config") else None
            
            if config:
                if hasattr(config, "hidden_size"):
                    batch_size = 1
                    seq_len = 16
                    hidden_size = config.hidden_size
                    sample_inputs["hidden_states"] = torch.rand(batch_size, seq_len, hidden_size)
                
                if hasattr(config, "vocab_size"):
                    batch_size = 1
                    seq_len = 16
                    sample_inputs["input_ids"] = torch.randint(0, config.vocab_size, (batch_size, seq_len))
                    sample_inputs["attention_mask"] = torch.ones(batch_size, seq_len, dtype=torch.long)
            
            # If we couldn't create any inputs, use a generic input
            if not sample_inputs:
                sample_inputs["input"] = torch.rand(1, 3, 224, 224)  # Common image input shape
                
            return sample_inputs
                
        except Exception as e:
            logger.warning(f"Failed to create sample inputs: {str(e)}")
            # Return a generic input as fallback
            return {"input": torch.rand(1, 3, 224, 224)}  # Common image input shape
