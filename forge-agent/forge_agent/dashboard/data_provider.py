"""
Data provider for the analysis and reporting dashboard.
"""
import datetime
from typing import Dict, List, Optional, Any

import pandas as pd
from loguru import logger

from forge_agent.dashboard.models import ChartData, DashboardData
from forge_agent.result_tracking.database import ResultTrackingDatabase
from forge_agent.result_tracking.models import TestResultFilter, TestResultStatistics
from forge_agent.test_pipeline.models import TestStatus, AdaptationLevel, FailureReason


class DashboardDataProvider:
    """Provides data for the analysis and reporting dashboard."""
    
    def __init__(self, result_db: ResultTrackingDatabase):
        """
        Initialize the dashboard data provider.
        
        Args:
            result_db: Result tracking database instance
        """
        self.result_db = result_db
    
    def get_dashboard_data(self, timeframe_days: int = 7) -> DashboardData:
        """
        Get data for the dashboard.
        
        Args:
            timeframe_days: Timeframe to include in days
            
        Returns:
            DashboardData object with dashboard information
        """
        # Calculate the start date for the timeframe
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=timeframe_days)
        
        # Create filter for the timeframe
        filter_criteria = TestResultFilter(
            start_date=start_date,
            end_date=end_date
        )
        
        # Get statistics for the timeframe
        statistics = self.result_db.get_statistics(filter_criteria)
        
        # Get recent test results
        recent_results = self.result_db.filter_test_results(
            TestResultFilter(
                limit=10,  # Show only the 10 most recent results
                start_date=start_date,
                end_date=end_date
            )
        )
        recent_results_data = [
            {
                "id": result.id,
                "model_id": result.model_id,
                "status": result.status.value,
                "timestamp": result.timestamp.isoformat(),
                "adaptation_level": result.adaptation_level.value if result.adaptation_level else None,
                "failure_reason": result.failure_reason.value if result.failure_reason else None
            }
            for result in recent_results
        ]
        
        # Create charts
        charts = [
            self._create_status_chart(statistics),
            self._create_adaptation_level_chart(statistics),
            self._create_failure_reason_chart(statistics),
            self._create_model_performance_chart(filter_criteria)
        ]
        
        # Create summary metrics
        summary_metrics = {
            "total_tests": statistics.total_count,
            "success_rate": f"{statistics.success_rate * 100:.1f}%",
            "avg_compilation_time": f"{statistics.avg_compilation_time_seconds:.2f}s" if statistics.avg_compilation_time_seconds else "N/A",
            "avg_execution_time": f"{statistics.avg_execution_time_seconds:.2f}s" if statistics.avg_execution_time_seconds else "N/A",
            "avg_memory_usage": f"{statistics.avg_memory_usage_mb:.1f}MB" if statistics.avg_memory_usage_mb else "N/A"
        }
        
        # Create dashboard data
        return DashboardData(
            title=f"Hugging Face Model Compatibility Dashboard (Last {timeframe_days} Days)",
            summary_metrics=summary_metrics,
            charts=charts,
            recent_test_results=recent_results_data,
            timeframe_days=timeframe_days
        )
    
    def _create_status_chart(self, statistics: TestResultStatistics) -> ChartData:
        """Create chart for test status distribution."""
        labels = []
        data = []
        
        for status in TestStatus:
            count = statistics.status_counts.get(status.value, 0)
            if count > 0:
                labels.append(status.value)
                data.append(count)
        
        return ChartData(
            chart_id="status_chart",
            chart_type="pie",
            title="Test Status Distribution",
            labels=labels,
            datasets=[{
                "data": data,
                "backgroundColor": [
                    "#4CAF50",  # COMPLETED - Green
                    "#FFC107",  # QUEUED - Yellow
                    "#2196F3",  # DOWNLOADING - Blue
                    "#9C27B0",  # ADAPTING - Purple
                    "#FF5722",  # COMPILING - Orange
                    "#03A9F4",  # EXECUTING - Light Blue
                    "#F44336"   # FAILED - Red
                ]
            }],
            options={
                "responsive": True,
                "maintainAspectRatio": False
            }
        )
    
    def _create_adaptation_level_chart(self, statistics: TestResultStatistics) -> ChartData:
        """Create chart for adaptation level distribution."""
        labels = []
        data = []
        
        for level in AdaptationLevel:
            count = statistics.adaptation_level_counts.get(level.value, 0)
            if count > 0:
                labels.append(level.value)
                data.append(count)
        
        return ChartData(
            chart_id="adaptation_level_chart",
            chart_type="bar",
            title="Adaptation Level Distribution",
            labels=labels,
            datasets=[{
                "label": "Count",
                "data": data,
                "backgroundColor": "#3F51B5"
            }],
            options={
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "y": {
                        "beginAtZero": True
                    }
                }
            }
        )
    
    def _create_failure_reason_chart(self, statistics: TestResultStatistics) -> ChartData:
        """Create chart for failure reason distribution."""
        labels = []
        data = []
        
        for reason in FailureReason:
            count = statistics.failure_reason_counts.get(reason.value, 0)
            if count > 0:
                labels.append(reason.value)
                data.append(count)
        
        return ChartData(
            chart_id="failure_reason_chart",
            chart_type="bar",
            title="Failure Reason Distribution",
            labels=labels,
            datasets=[{
                "label": "Count",
                "data": data,
                "backgroundColor": "#E91E63"
            }],
            options={
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "y": {
                        "beginAtZero": True
                    }
                }
            }
        )
    
    def _create_model_performance_chart(self, filter_criteria: TestResultFilter) -> ChartData:
        """Create chart for model performance metrics."""
        # Get test results for the filter criteria
        df = self.result_db.export_to_dataframe(filter_criteria)
        
        # Filter for completed tests only
        df = df[df['status'] == TestStatus.COMPLETED.value]
        
        # Get top models by download count (limited to N)
        top_n = 10
        if len(df) > top_n:
            df = df.nlargest(top_n, 'compilation_time_seconds')
        
        model_ids = df['model_id'].tolist()
        compilation_times = df['compilation_time_seconds'].tolist()
        execution_times = df['execution_time_seconds'].tolist()
        
        return ChartData(
            chart_id="model_performance_chart",
            chart_type="bar",
            title=f"Performance of Top {len(model_ids)} Models",
            labels=model_ids,
            datasets=[
                {
                    "label": "Compilation Time (s)",
                    "data": compilation_times,
                    "backgroundColor": "#2196F3"
                },
                {
                    "label": "Execution Time (s)",
                    "data": execution_times,
                    "backgroundColor": "#FF9800"
                }
            ],
            options={
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Time (seconds)"
                        }
                    }
                }
            }
        )
