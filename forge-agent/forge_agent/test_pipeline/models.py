"""
Data models for the test pipeline.
"""
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TestStatus(str, Enum):
    """Status of a model test."""
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    ADAPTING = "adapting"
    COMPILING = "compiling"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class AdaptationLevel(str, Enum):
    """Level of adaptation required for a model."""
    NONE = "none"            # No adaptation needed
    LEVEL_1 = "level_1"      # Low effort - standard templates and minimal changes
    LEVEL_2 = "level_2"      # Medium effort - moderate code changes
    LEVEL_3 = "level_3"      # High effort - significant restructuring
    BEYOND = "beyond"        # Beyond current capability


class FailureReason(str, Enum):
    """Common failure reasons for model tests."""
    DOWNLOAD_ERROR = "download_error"
    ADAPTATION_ERROR = "adaptation_error"
    COMPILATION_ERROR = "compilation_error"
    EXECUTION_ERROR = "execution_error"
    UNSUPPORTED_ARCHITECTURE = "unsupported_architecture"
    MEMORY_LIMIT = "memory_limit"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class TestConfig(BaseModel):
    """Configuration for a model test."""
    model_id: str = Field(..., description="Hugging Face model ID")
    timeout_seconds: int = Field(3600, description="Timeout for the entire test in seconds")
    use_llm_adaptation: bool = Field(True, description="Whether to use LLM for adaptation if needed")
    sample_input_shape: Dict[str, List[int]] = Field(default_factory=dict, 
                                                     description="Shape of sample inputs, keyed by input name")
    compiler_config: Dict[str, Any] = Field(default_factory=dict,
                                            description="Configuration for the tt-torch compiler")


class TestResult(BaseModel):
    """Results of a model test."""
    model_id: str = Field(..., description="Hugging Face model ID")
    status: TestStatus = Field(TestStatus.QUEUED, description="Current status of the test")
    adaptation_level: Optional[AdaptationLevel] = Field(None, description="Level of adaptation required")
    failure_reason: Optional[FailureReason] = Field(None, description="Reason for failure if test failed")
    error_message: Optional[str] = Field(None, description="Detailed error message")
    compilation_time_seconds: Optional[float] = Field(None, description="Time taken for compilation")
    execution_time_seconds: Optional[float] = Field(None, description="Time taken for execution")
    memory_usage_mb: Optional[float] = Field(None, description="Peak memory usage during testing")
    adaptation_code: Optional[str] = Field(None, description="Code used for adaptation")
    output_matches_reference: Optional[bool] = Field(None, 
                                                     description="Whether outputs match reference within tolerance")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Additional metrics collected during testing")
