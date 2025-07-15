"""
Client for interacting with the Hugging Face API to fetch model data.
"""
import os
from typing import Dict, List, Optional, Any

import requests
from huggingface_hub import HfApi
from loguru import logger

from forge_agent.metadata_service.models import ModelMetadata, ModelArchitecture, ModelFramework


class HuggingFaceClient:
    """Client for interacting with the Hugging Face API."""

    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize the Hugging Face client.

        Args:
            api_token: Optional Hugging Face API token for authenticated requests.
                       If not provided, will try to use HF_API_TOKEN environment variable.
        """
        self.api_token = api_token or os.environ.get("HF_API_TOKEN")
        self.hf_api = HfApi(token=self.api_token)
        self.base_url = "https://huggingface.co/api"

    def get_top_models(self, limit: int = 10000, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the top models from Hugging Face by download count.

        Args:
            limit: Maximum number of models to return
            model_type: Optional filter for model type

        Returns:
            List of model data dictionaries
        """
        logger.info(f"Fetching top {limit} models from Hugging Face")
        
        # The HF API doesn't directly allow sorting by downloads, so we need to fetch models
        # and then sort them manually
        models = []
        cursor = None
        page_size = 500  # HF API page size
        
        # Parameters for the API call
        params = {"limit": page_size}
        if model_type:
            params["filter"] = model_type
            
        headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
        
        while len(models) < limit:
            if cursor:
                params["cursor"] = cursor
                
            response = requests.get(
                f"{self.base_url}/models", 
                params=params,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch models: {response.text}")
                break
                
            data = response.json()
            models.extend(data)
            
            # Check if there are more results
            cursor = response.headers.get("X-Cursor")
            if not cursor or len(data) < page_size:
                break
                
        # Sort by downloads (descending) and take the top 'limit' models
        models.sort(key=lambda x: x.get("downloads", 0), reverse=True)
        return models[:limit]

    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """
        Get detailed metadata for a specific model.

        Args:
            model_id: Hugging Face model ID (e.g. 'bert-base-uncased')

        Returns:
            ModelMetadata object with model information
        """
        logger.info(f"Fetching metadata for model: {model_id}")
        
        try:
            # Get model info from the API
            model_info = self.hf_api.model_info(model_id)
            
            # Determine the framework
            framework = self._determine_framework(model_info)
            
            # Determine the architecture
            architecture = self._determine_architecture(model_info, model_id)
            
            # Extract size in MB (if available)
            size_mb = None
            if hasattr(model_info, 'siblings') and model_info.siblings:
                total_size = sum(s.size for s in model_info.siblings if hasattr(s, 'size'))
                size_mb = total_size / (1024 * 1024)  # Convert to MB
            
            # Create and return model metadata
            metadata = ModelMetadata(
                model_id=model_id,
                framework=framework,
                architecture=architecture,
                downloads=getattr(model_info, 'downloads', 0),
                tags=getattr(model_info, 'tags', []),
                size_mb=size_mb,
                last_modified=str(getattr(model_info, 'last_modified', None)),
                raw_metadata={
                    key: value for key, value in model_info.__dict__.items()
                    if not key.startswith('_') and key != 'siblings'  # Exclude large attributes
                }
            )
            return metadata
            
        except Exception as e:
            logger.error(f"Error fetching metadata for {model_id}: {str(e)}")
            # Return basic metadata with error info
            return ModelMetadata(
                model_id=model_id,
                raw_metadata={"error": str(e)}
            )
    
    def _determine_framework(self, model_info: Any) -> ModelFramework:
        """Determine the framework used by the model."""
        # Check the model's library_name attribute
        library = getattr(model_info, 'library_name', '').lower()
        
        if 'pytorch' in library or 'torch' in library:
            return ModelFramework.PYTORCH
        elif 'tensorflow' in library or 'tf' in library:
            return ModelFramework.TENSORFLOW
        elif 'jax' in library or 'flax' in library:
            return ModelFramework.JAX
        
        # Check tags
        tags = getattr(model_info, 'tags', [])
        if any(tag in ['pytorch', 'torch'] for tag in tags):
            return ModelFramework.PYTORCH
        elif any(tag in ['tensorflow', 'tf'] for tag in tags):
            return ModelFramework.TENSORFLOW
        elif any(tag in ['jax', 'flax'] for tag in tags):
            return ModelFramework.JAX
            
        return ModelFramework.UNKNOWN
        
    def _determine_architecture(self, model_info: Any, model_id: str) -> ModelArchitecture:
        """Determine the model architecture."""
        # Check the model ID first
        model_id_lower = model_id.lower()
        
        for arch in ModelArchitecture:
            # Skip the 'OTHER' type
            if arch == ModelArchitecture.OTHER:
                continue
                
            # Check if the architecture name is in the model ID
            if arch.value in model_id_lower:
                return arch
        
        # Check tags
        tags = getattr(model_info, 'tags', [])
        for tag in tags:
            for arch in ModelArchitecture:
                if arch != ModelArchitecture.OTHER and arch.value in tag.lower():
                    return arch
        
        return ModelArchitecture.OTHER
