"""
Data models for the analysis and reporting dashboard.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DashboardConfig(BaseModel):
    """Configuration for the dashboard."""
    title: str = Field("Hugging Face Model Compatibility Dashboard", description="Dashboard title")
    refresh_interval_seconds: int = Field(60, description="Dashboard refresh interval in seconds")
    default_timeframe_days: int = Field(7, description="Default timeframe to display in days")
    theme: str = Field("light", description="Dashboard theme (light or dark)")
    max_models_in_charts: int = Field(20, description="Maximum number of models to show in charts")


class ModelPerformanceMetrics(BaseModel):
    """Performance metrics for a model."""
    model_id: str = Field(..., description="Hugging Face model ID")
    compilation_time_seconds: Optional[float] = Field(None, description="Time taken for compilation")
    execution_time_seconds: Optional[float] = Field(None, description="Time taken for execution")
    memory_usage_mb: Optional[float] = Field(None, description="Peak memory usage during testing")
    custom_metrics: Dict[str, Any] = Field(default_factory=dict, description="Custom performance metrics")


class ChartData(BaseModel):
    """Data for a chart in the dashboard."""
    chart_id: str = Field(..., description="Unique identifier for the chart")
    chart_type: str = Field(..., description="Type of chart (bar, line, pie, etc.)")
    title: str = Field(..., description="Chart title")
    labels: List[str] = Field(..., description="Labels for the chart data")
    datasets: List[Dict[str, Any]] = Field(..., description="Datasets for the chart")
    options: Dict[str, Any] = Field(default_factory=dict, description="Chart options")


class DashboardData(BaseModel):
    """Data for the dashboard."""
    title: str = Field("Hugging Face Model Compatibility Dashboard", description="Dashboard title")
    summary_metrics: Dict[str, Any] = Field(..., description="Summary metrics for the dashboard")
    charts: List[ChartData] = Field(..., description="Charts for the dashboard")
    recent_test_results: List[Dict[str, Any]] = Field(..., description="Recent test results")
    timeframe_days: int = Field(7, description="Timeframe displayed in days")
