"""
Models for the Hugging Face model metadata service.
"""
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ModelFramework(str, Enum):
    """Supported frameworks for models."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    UNKNOWN = "unknown"


class ModelArchitecture(str, Enum):
    """Common model architectures."""
    BERT = "bert"
    GPT2 = "gpt2"
    T5 = "t5"
    BART = "bart"
    LLAMA = "llama"
    ROBERTA = "roberta"
    DISTILBERT = "distilbert"
    VIT = "vit"
    RESNET = "resnet"
    OTHER = "other"


class ModelMetadata(BaseModel):
    """Metadata for a Hugging Face model."""
    model_id: str = Field(..., description="The Hugging Face model ID (e.g. 'bert-base-uncased')")
    framework: ModelFramework = Field(ModelFramework.UNKNOWN, description="Framework the model uses")
    architecture: ModelArchitecture = Field(ModelArchitecture.OTHER, description="Model architecture type")
    downloads: int = Field(0, description="Download count from Hugging Face")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the model")
    size_mb: Optional[float] = Field(None, description="Size of the model in MB")
    last_modified: Optional[str] = Field(None, description="Last modified timestamp")
    is_compatible: Optional[bool] = Field(None, description="Whether the model is compatible with tt-torch")
    raw_metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw metadata from Hugging Face API")

    class Config:
        """Pydantic model configuration."""
        frozen = False


class ModelSelectionCriteria(BaseModel):
    """Criteria for selecting models to test."""
    min_downloads: Optional[int] = Field(None, description="Minimum download count")
    max_size_mb: Optional[float] = Field(None, description="Maximum model size in MB")
    frameworks: List[ModelFramework] = Field(default_factory=list, description="Frameworks to include")
    architectures: List[ModelArchitecture] = Field(default_factory=list, description="Architectures to include")
    tags: List[str] = Field(default_factory=list, description="Tags to filter by")
    limit: int = Field(10000, description="Maximum number of models to select")
