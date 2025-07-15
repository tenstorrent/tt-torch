"""
Data models for the result tracking database.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from forge_agent.test_pipeline.models import TestStatus, AdaptationLevel, FailureReason


class TestRecord(BaseModel):
    """Record of a model test result for persistence."""
    id: Optional[int] = Field(None, description="Database ID")
    model_id: str = Field(..., description="Hugging Face model ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the test")
    status: TestStatus = Field(TestStatus.QUEUED, description="Status of the test")
    adaptation_level: Optional[AdaptationLevel] = Field(None, description="Level of adaptation required")
    failure_reason: Optional[FailureReason] = Field(None, description="Reason for failure if test failed")
    error_message: Optional[str] = Field(None, description="Detailed error message")
    stack_trace: Optional[str] = Field(None, description="Stack trace if an error occurred")
    compilation_time_seconds: Optional[float] = Field(None, description="Time taken for compilation")
    execution_time_seconds: Optional[float] = Field(None, description="Time taken for execution")
    memory_usage_mb: Optional[float] = Field(None, description="Peak memory usage during testing")
    adaptation_code: Optional[str] = Field(None, description="Code used for adaptation")
    output_matches_reference: Optional[bool] = Field(None, 
                                                     description="Whether outputs match reference within tolerance")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Additional metrics collected during testing")
    tags: List[str] = Field(default_factory=list, description="Tags associated with this test")


class TestResultFilter(BaseModel):
    """Filter criteria for querying test results."""
    model_ids: Optional[List[str]] = Field(None, description="List of model IDs to filter by")
    statuses: Optional[List[TestStatus]] = Field(None, description="List of statuses to filter by")
    adaptation_levels: Optional[List[AdaptationLevel]] = Field(None, description="List of adaptation levels to filter by")
    failure_reasons: Optional[List[FailureReason]] = Field(None, description="List of failure reasons to filter by")
    tags: Optional[List[str]] = Field(None, description="List of tags to filter by")
    start_date: Optional[datetime] = Field(None, description="Filter tests after this date")
    end_date: Optional[datetime] = Field(None, description="Filter tests before this date")
    limit: Optional[int] = Field(None, description="Maximum number of results to return")
    offset: Optional[int] = Field(None, description="Offset for pagination")


class TestResultStatistics(BaseModel):
    """Statistics about test results."""
    total_count: int = Field(0, description="Total number of tests")
    status_counts: Dict[str, int] = Field(default_factory=dict, description="Count of tests by status")
    adaptation_level_counts: Dict[str, int] = Field(default_factory=dict, description="Count of tests by adaptation level")
    failure_reason_counts: Dict[str, int] = Field(default_factory=dict, description="Count of tests by failure reason")
    avg_compilation_time_seconds: Optional[float] = Field(None, description="Average compilation time")
    avg_execution_time_seconds: Optional[float] = Field(None, description="Average execution time")
    avg_memory_usage_mb: Optional[float] = Field(None, description="Average memory usage")
    success_rate: float = Field(0.0, description="Percentage of tests that completed successfully")
