"""
Module for downloading models from Hugging Face.
"""
import os
from typing import Dict, Optional, Tuple, Any

from huggingface_hub import snapshot_download, hf_hub_download
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
from loguru import logger
import requests
from PIL import Image
from torchvision import transforms

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
    
    def _create_real_image_input(self, model_id: str) -> Dict[str, torch.Tensor]:
        """
        Create real image input for vision models (like ResNet) using actual images.
        Uses the same approach as the ResNet demo.
        
        Args:
            model_id: Hugging Face model ID
            
        Returns:
            Dictionary with real image tensor
        """
        try:
            # Use the same default image as the ResNet demo
            DEFAULT_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
            
            logger.info(f"Downloading real image for {model_id} from COCO dataset")
            logger.info(f"Image URL: {DEFAULT_URL}")
            
            # Download and open the image
            response = requests.get(DEFAULT_URL, stream=True, timeout=10)
            response.raise_for_status()
            img = Image.open(response.raw)
            
            # Create standard ImageNet preprocessing transforms
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Preprocess the image
            img_tensor = preprocess(img)
            img_tensor = torch.unsqueeze(img_tensor, 0)  # Add batch dimension
            
            logger.info(f"Real image processed for {model_id}: shape {img_tensor.shape}")
            logger.info(f"Image contains: Two cats on a couch (COCO dataset)")
            
            return {"pixel_values": img_tensor}
            
        except Exception as e:
            logger.warning(f"Failed to download real image for {model_id}: {str(e)}")
            logger.warning(f"Falling back to random tensor")
            # Fallback to random tensor if image download fails
            return {"pixel_values": torch.rand(1, 3, 224, 224)}
    
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
                    # If model is encoder-decoder, avoid conflicting decoder inputs
                    try:
                        if hasattr(model, "config") and getattr(model.config, "is_encoder_decoder", False):
                            # Ensure we don't include decoder_inputs_embeds from tokenizer outputs
                            if isinstance(inputs, dict) and "decoder_inputs_embeds" in inputs:
                                inputs.pop("decoder_inputs_embeds", None)
                            # Add minimal decoder_input_ids if not present
                            if isinstance(inputs, dict) and "decoder_input_ids" not in inputs:
                                bos_id = getattr(model.config, "decoder_start_token_id", None) or getattr(model.config, "bos_token_id", None) or 1
                                import torch as _torch
                                inputs["decoder_input_ids"] = _torch.tensor([[bos_id]], dtype=_torch.long)
                    except Exception as _e:
                        logger.debug(f"Seq2seq input sanitation skipped: {_e}")
                    sample_inputs = inputs
                    return sample_inputs
            
            # For other model types, try to infer from config and model type
            config = model.config if hasattr(model, "config") else None
            model_class_name = model.__class__.__name__.lower()
            
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
            
            # If we couldn't create any inputs, use model-specific defaults
            if not sample_inputs:
                # Check model type for appropriate input parameter names
                if "resnet" in model_class_name or "vit" in model_class_name or "efficientnet" in model_class_name:
                    # For ResNet models, use real images like the demo
                    if "resnet" in model_class_name:
                        logger.info(f"Creating REAL IMAGE input for ResNet model: {model_class_name}")
                        sample_inputs = self._create_real_image_input(test_config.model_id)
                        logger.debug(f"Created real image inputs for {model_class_name}: {list(sample_inputs.keys())}")
                    else:
                        # For other vision models, use random tensors
                        sample_inputs["pixel_values"] = torch.rand(1, 3, 224, 224)
                        logger.debug(f"Created random vision model inputs for {model_class_name}: pixel_values")
                elif "gpt" in model_class_name or "llama" in model_class_name or "bert" in model_class_name or "m2m100" in model_class_name or "t5" in model_class_name or "bart" in model_class_name:
                    # Language models typically use "input_ids"
                    sample_inputs["input_ids"] = torch.randint(0, 1000, (1, 16))
                    sample_inputs["attention_mask"] = torch.ones(1, 16, dtype=torch.long)
                    # For encoder-decoder families add minimal decoder_input_ids
                    try:
                        if hasattr(model, "config") and getattr(model.config, "is_encoder_decoder", False):
                            bos_id = getattr(model.config, "decoder_start_token_id", None) or getattr(model.config, "bos_token_id", None) or 1
                            sample_inputs["decoder_input_ids"] = torch.tensor([[bos_id]], dtype=torch.long)
                    except Exception as _e:
                        logger.debug(f"Adding decoder_input_ids skipped: {_e}")
                    logger.debug(f"Created language model inputs for {model_class_name}: input_ids, attention_mask")
                else:
                    # Generic fallback - try both common parameter names
                    sample_inputs["pixel_values"] = torch.rand(1, 3, 224, 224)
                    logger.debug(f"Created generic vision inputs for {model_class_name}: pixel_values")
                
            logger.info(f"Sample inputs created for {test_config.model_id}: {list(sample_inputs.keys())}")
            return sample_inputs
                
        except Exception as e:
            logger.warning(f"Failed to create sample inputs: {str(e)}")
            # Return a generic input as fallback - use pixel_values for vision models
            return {"pixel_values": torch.rand(1, 3, 224, 224)}  # Common image input shape
