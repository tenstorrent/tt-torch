"""
API for the analysis and reporting dashboard.
"""
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, Depends, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from loguru import logger

from forge_agent.result_tracking.database import ResultTrackingDatabase
from forge_agent.result_tracking.models import TestResultFilter
from forge_agent.test_pipeline.models import TestStatus, AdaptationLevel, FailureReason
from forge_agent.dashboard.data_provider import DashboardDataProvider


# Create the FastAPI app
app = FastAPI(
    title="Hugging Face Model Compatibility Dashboard",
    description="Dashboard for tracking model compatibility with tt-torch",
    version="0.1.0"
)


# Dependency to get the database
def get_result_db():
    """Get the result tracking database."""
    db = ResultTrackingDatabase()
    try:
        yield db
    finally:
        pass  # Database session management is handled internally


@app.get("/api/dashboard", tags=["Dashboard"])
async def get_dashboard_data(
    timeframe_days: int = Query(7, description="Timeframe in days"),
    result_db: ResultTrackingDatabase = Depends(get_result_db)
):
    """Get dashboard data."""
    try:
        data_provider = DashboardDataProvider(result_db)
        return data_provider.get_dashboard_data(timeframe_days=timeframe_days)
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        return {"error": str(e)}


@app.get("/api/models", tags=["Models"])
async def get_models(
    status: Optional[str] = None,
    adaptation_level: Optional[str] = None,
    limit: int = Query(100, description="Maximum number of models to return"),
    offset: int = Query(0, description="Offset for pagination"),
    result_db: ResultTrackingDatabase = Depends(get_result_db)
):
    """Get model test results with optional filtering."""
    try:
        # Create filter criteria
        filter_criteria = TestResultFilter(
            limit=limit,
            offset=offset
        )
        
        # Add status filter if provided
        if status:
            try:
                filter_criteria.statuses = [TestStatus(status)]
            except ValueError:
                return {"error": f"Invalid status: {status}"}
        
        # Add adaptation level filter if provided
        if adaptation_level:
            try:
                filter_criteria.adaptation_levels = [AdaptationLevel(adaptation_level)]
            except ValueError:
                return {"error": f"Invalid adaptation level: {adaptation_level}"}
        
        # Get filtered results
        results = result_db.filter_test_results(filter_criteria)
        
        # Convert to JSON-serializable format
        return [
            {
                "id": r.id,
                "model_id": r.model_id,
                "timestamp": r.timestamp.isoformat(),
                "status": r.status.value,
                "adaptation_level": r.adaptation_level.value if (r.adaptation_level and hasattr(r.adaptation_level, 'value')) else str(r.adaptation_level) if r.adaptation_level else None,
                "failure_reason": r.failure_reason.value if r.failure_reason else None,
                "compilation_time_seconds": r.compilation_time_seconds,
                "execution_time_seconds": r.execution_time_seconds,
                "memory_usage_mb": r.memory_usage_mb
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return {"error": str(e)}


@app.get("/api/model/{model_id}", tags=["Models"])
async def get_model_details(
    model_id: str,
    result_db: ResultTrackingDatabase = Depends(get_result_db)
):
    """Get detailed information about a specific model."""
    try:
        # Get the latest result for the model
        result = result_db.get_latest_result_for_model(model_id)
        
        if not result:
            return {"error": f"Model {model_id} not found"}
        
        # Convert to JSON-serializable format
        return {
            "id": result.id,
            "model_id": result.model_id,
            "timestamp": result.timestamp.isoformat(),
            "status": result.status.value,
            "adaptation_level": result.adaptation_level.value if (result.adaptation_level and hasattr(result.adaptation_level, 'value')) else str(result.adaptation_level) if result.adaptation_level else None,
            "failure_reason": result.failure_reason.value if result.failure_reason else None,
            "error_message": result.error_message,
            "compilation_time_seconds": result.compilation_time_seconds,
            "execution_time_seconds": result.execution_time_seconds,
            "memory_usage_mb": result.memory_usage_mb,
            "output_matches_reference": result.output_matches_reference,
            "metrics": result.metrics,
            "tags": result.tags
        }
    except Exception as e:
        logger.error(f"Error getting model details: {str(e)}")
        return {"error": str(e)}


@app.get("/api/statistics", tags=["Statistics"])
async def get_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    result_db: ResultTrackingDatabase = Depends(get_result_db)
):
    """Get statistics about model tests."""
    try:
        # Create filter criteria
        filter_criteria = TestResultFilter()
        
        # Parse dates if provided
        if start_date:
            try:
                filter_criteria.start_date = datetime.fromisoformat(start_date)
            except ValueError:
                return {"error": f"Invalid start date format: {start_date}"}
        
        if end_date:
            try:
                filter_criteria.end_date = datetime.fromisoformat(end_date)
            except ValueError:
                return {"error": f"Invalid end date format: {end_date}"}
        
        # Get statistics
        statistics = result_db.get_statistics(filter_criteria)
        
        # Convert to JSON-serializable format
        return {
            "total_count": statistics.total_count,
            "status_counts": statistics.status_counts,
            "adaptation_level_counts": statistics.adaptation_level_counts,
            "failure_reason_counts": statistics.failure_reason_counts,
            "avg_compilation_time_seconds": statistics.avg_compilation_time_seconds,
            "avg_execution_time_seconds": statistics.avg_execution_time_seconds,
            "avg_memory_usage_mb": statistics.avg_memory_usage_mb,
            "success_rate": statistics.success_rate
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def get_dashboard():
    """Serve the dashboard HTML page."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Hugging Face Model Compatibility Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #3498db;
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
        .summary {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            flex: 1;
            min-width: 150px;
        }
        .metric-title {
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .charts {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .chart-container {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            height: 300px;
        }
        .chart-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .table-container {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        .timeframe-selector {
            margin-bottom: 20px;
            text-align: right;
        }
        select {
            padding: 8px 12px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .status-badge {
            padding: 5px 10px;
            border-radius: 4px;
            color: white;
            font-size: 12px;
        }
        .status-completed { background-color: #2ecc71; }
        .status-failed { background-color: #e74c3c; }
        .status-queued { background-color: #f39c12; }
        .status-running { background-color: #3498db; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Hugging Face Model Compatibility Dashboard</h1>
    </div>
    
    <div class="container">
        <div class="timeframe-selector">
            <label for="timeframe">Timeframe:</label>
            <select id="timeframe" onchange="loadDashboardData()">
                <option value="7">Last 7 Days</option>
                <option value="30">Last 30 Days</option>
                <option value="90">Last 90 Days</option>
                <option value="365">Last Year</option>
                <option value="99999">All Time</option>
            </select>
        </div>
        
        <div class="summary" id="summary">
            <!-- Summary metrics will be inserted here -->
        </div>
        
        <div class="charts">
            <div class="chart-container">
                <div class="chart-title">Test Status Distribution</div>
                <canvas id="statusChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Adaptation Level Distribution</div>
                <canvas id="adaptationLevelChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Failure Reason Distribution</div>
                <canvas id="failureReasonChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Model Performance</div>
                <canvas id="modelPerformanceChart"></canvas>
            </div>
        </div>
        
        <div class="table-container">
            <h2>Recent Test Results</h2>
            <table id="recentResultsTable">
                <thead>
                    <tr>
                        <th>Model ID</th>
                        <th>Status</th>
                        <th>Adaptation Level</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody id="recentResultsBody">
                    <!-- Table rows will be inserted here -->
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        let charts = {};
        
        // Load dashboard data
        function loadDashboardData() {
            const timeframe = document.getElementById('timeframe').value;
            axios.get(`/api/dashboard?timeframe_days=${timeframe}`)
                .then(response => {
                    const dashboardData = response.data;
                    updateSummaryMetrics(dashboardData.summary_metrics);
                    updateCharts(dashboardData.charts);
                    updateRecentResults(dashboardData.recent_test_results);
                })
                .catch(error => {
                    console.error('Error loading dashboard data:', error);
                });
        }
        
        // Update summary metrics
        function updateSummaryMetrics(metrics) {
            const summaryContainer = document.getElementById('summary');
            summaryContainer.innerHTML = '';
            
            for (const [key, value] of Object.entries(metrics)) {
                const metricCard = document.createElement('div');
                metricCard.className = 'metric-card';
                
                const title = document.createElement('div');
                title.className = 'metric-title';
                title.textContent = formatMetricName(key);
                
                const valueElem = document.createElement('div');
                valueElem.className = 'metric-value';
                valueElem.textContent = value;
                
                metricCard.appendChild(title);
                metricCard.appendChild(valueElem);
                summaryContainer.appendChild(metricCard);
            }
        }
        
        // Format metric name
        function formatMetricName(name) {
            return name
                .replace(/_/g, ' ')
                .replace(/\\b\\w/g, l => l.toUpperCase());
        }
        
        // Update charts
        function updateCharts(chartsData) {
            for (const chartData of chartsData) {
                if (charts[chartData.chart_id]) {
                    charts[chartData.chart_id].destroy();
                }
                
                const ctx = document.getElementById(chartData.chart_id).getContext('2d');
                charts[chartData.chart_id] = new Chart(ctx, {
                    type: chartData.chart_type,
                    data: {
                        labels: chartData.labels,
                        datasets: chartData.datasets
                    },
                    options: chartData.options
                });
            }
        }
        
        // Update recent results table
        function updateRecentResults(results) {
            const tableBody = document.getElementById('recentResultsBody');
            tableBody.innerHTML = '';
            
            for (const result of results) {
                const row = document.createElement('tr');
                
                const modelIdCell = document.createElement('td');
                modelIdCell.textContent = result.model_id;
                
                const statusCell = document.createElement('td');
                const statusBadge = document.createElement('span');
                statusBadge.className = `status-badge status-${result.status.toLowerCase()}`;
                statusBadge.textContent = result.status;
                statusCell.appendChild(statusBadge);
                
                const adaptationLevelCell = document.createElement('td');
                adaptationLevelCell.textContent = result.adaptation_level || 'N/A';
                
                const timestampCell = document.createElement('td');
                timestampCell.textContent = new Date(result.timestamp).toLocaleString();
                
                row.appendChild(modelIdCell);
                row.appendChild(statusCell);
                row.appendChild(adaptationLevelCell);
                row.appendChild(timestampCell);
                
                tableBody.appendChild(row);
            }
        }
        
        // Initial load
        document.addEventListener('DOMContentLoaded', loadDashboardData);
    </script>
</body>
</html>
    """
