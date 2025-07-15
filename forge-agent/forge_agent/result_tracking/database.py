"""
Database service for storing and querying test results.
"""
from typing import List, Optional, Dict, Any, Tuple
import os
from contextlib import contextmanager
from datetime import datetime
import json

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, 
    JSON, DateTime, Text, desc, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from loguru import logger

from forge_agent.result_tracking.models import (
    TestRecord, 
    TestResultFilter, 
    TestResultStatistics
)
from forge_agent.test_pipeline.models import TestStatus, AdaptationLevel, FailureReason

Base = declarative_base()


class TestResultTable(Base):
    """SQLAlchemy model for test results."""
    __tablename__ = "test_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    status = Column(String, nullable=False, index=True)
    adaptation_level = Column(String, nullable=True, index=True)
    failure_reason = Column(String, nullable=True, index=True)
    error_message = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)
    compilation_time_seconds = Column(Float, nullable=True)
    execution_time_seconds = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    adaptation_code = Column(Text, nullable=True)
    output_matches_reference = Column(Boolean, nullable=True)
    metrics = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=False)


class ResultTrackingDatabase:
    """Database service for test results."""

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize the result tracking database.

        Args:
            db_url: SQLAlchemy database URL. If None, uses a local SQLite database.
        """
        if db_url is None:
            # Default to SQLite database in the package directory
            package_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(package_dir, "..", "..", "data", "test_results.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            db_url = f"sqlite:///{db_path}"
            
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        logger.info(f"Initialized result tracking database at {db_url}")

    @contextmanager
    def session_scope(self):
        """Context manager for database sessions."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def store_test_result(self, test_record: TestRecord) -> int:
        """
        Store a test result in the database.

        Args:
            test_record: TestRecord to store

        Returns:
            ID of the stored record
        """
        try:
            with self.session_scope() as session:
                # Convert test record to database model
                db_record = TestResultTable(
                    model_id=test_record.model_id,
                    timestamp=test_record.timestamp,
                    status=test_record.status.value,
                    adaptation_level=test_record.adaptation_level.value if test_record.adaptation_level else None,
                    failure_reason=test_record.failure_reason.value if test_record.failure_reason else None,
                    error_message=test_record.error_message,
                    stack_trace=test_record.stack_trace,
                    compilation_time_seconds=test_record.compilation_time_seconds,
                    execution_time_seconds=test_record.execution_time_seconds,
                    memory_usage_mb=test_record.memory_usage_mb,
                    adaptation_code=test_record.adaptation_code,
                    output_matches_reference=test_record.output_matches_reference,
                    metrics=test_record.metrics,
                    tags=test_record.tags
                )
                
                session.add(db_record)
                session.flush()
                record_id = db_record.id
                
                logger.info(f"Stored test result for {test_record.model_id} with ID {record_id}")
                
                return record_id
                
        except Exception as e:
            logger.error(f"Error storing test result: {str(e)}")
            raise

    def get_test_result(self, result_id: int) -> Optional[TestRecord]:
        """
        Get a test result by ID.

        Args:
            result_id: ID of the test result

        Returns:
            TestRecord if found, None otherwise
        """
        try:
            with self.session_scope() as session:
                record = session.query(TestResultTable).filter_by(id=result_id).first()
                
                if record:
                    return self._convert_to_test_record(record)
                    
                return None
                
        except Exception as e:
            logger.error(f"Error getting test result: {str(e)}")
            return None

    def get_latest_result_for_model(self, model_id: str) -> Optional[TestRecord]:
        """
        Get the latest test result for a specific model.

        Args:
            model_id: Hugging Face model ID

        Returns:
            Latest TestRecord for the model, or None if not found
        """
        try:
            with self.session_scope() as session:
                record = session.query(TestResultTable)\
                    .filter_by(model_id=model_id)\
                    .order_by(desc(TestResultTable.timestamp))\
                    .first()
                
                if record:
                    return self._convert_to_test_record(record)
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting latest result for model {model_id}: {str(e)}")
            return None

    def filter_test_results(self, filter_criteria: TestResultFilter) -> List[TestRecord]:
        """
        Filter test results based on criteria.

        Args:
            filter_criteria: Criteria for filtering results

        Returns:
            List of TestRecord objects matching the criteria
        """
        try:
            with self.session_scope() as session:
                query = session.query(TestResultTable)
                
                # Apply filters
                if filter_criteria.model_ids:
                    query = query.filter(TestResultTable.model_id.in_(filter_criteria.model_ids))
                
                if filter_criteria.statuses:
                    status_values = [s.value for s in filter_criteria.statuses]
                    query = query.filter(TestResultTable.status.in_(status_values))
                
                if filter_criteria.adaptation_levels:
                    level_values = [l.value for l in filter_criteria.adaptation_levels]
                    query = query.filter(TestResultTable.adaptation_level.in_(level_values))
                
                if filter_criteria.failure_reasons:
                    reason_values = [r.value for r in filter_criteria.failure_reasons]
                    query = query.filter(TestResultTable.failure_reason.in_(reason_values))
                
                if filter_criteria.start_date:
                    query = query.filter(TestResultTable.timestamp >= filter_criteria.start_date)
                
                if filter_criteria.end_date:
                    query = query.filter(TestResultTable.timestamp <= filter_criteria.end_date)
                
                # Apply pagination
                if filter_criteria.offset:
                    query = query.offset(filter_criteria.offset)
                
                if filter_criteria.limit:
                    query = query.limit(filter_criteria.limit)
                
                # Sort by timestamp (newest first)
                query = query.order_by(desc(TestResultTable.timestamp))
                
                # Convert results to TestRecord objects
                return [self._convert_to_test_record(record) for record in query.all()]
                
        except Exception as e:
            logger.error(f"Error filtering test results: {str(e)}")
            return []

    def get_statistics(self, filter_criteria: Optional[TestResultFilter] = None) -> TestResultStatistics:
        """
        Get statistics about test results.

        Args:
            filter_criteria: Optional criteria to filter results before computing statistics

        Returns:
            TestResultStatistics object
        """
        try:
            with self.session_scope() as session:
                # Start with a base query
                base_query = session.query(TestResultTable)
                
                # Apply filters if provided
                if filter_criteria:
                    if filter_criteria.model_ids:
                        base_query = base_query.filter(TestResultTable.model_id.in_(filter_criteria.model_ids))
                    
                    if filter_criteria.statuses:
                        status_values = [s.value for s in filter_criteria.statuses]
                        base_query = base_query.filter(TestResultTable.status.in_(status_values))
                    
                    if filter_criteria.adaptation_levels:
                        level_values = [l.value for l in filter_criteria.adaptation_levels]
                        base_query = base_query.filter(TestResultTable.adaptation_level.in_(level_values))
                    
                    if filter_criteria.failure_reasons:
                        reason_values = [r.value for r in filter_criteria.failure_reasons]
                        base_query = base_query.filter(TestResultTable.failure_reason.in_(reason_values))
                    
                    if filter_criteria.start_date:
                        base_query = base_query.filter(TestResultTable.timestamp >= filter_criteria.start_date)
                    
                    if filter_criteria.end_date:
                        base_query = base_query.filter(TestResultTable.timestamp <= filter_criteria.end_date)
                
                # Total count
                total_count = base_query.count()
                
                # Status counts
                status_counts = {}
                for status in TestStatus:
                    count = base_query.filter(TestResultTable.status == status.value).count()
                    status_counts[status.value] = count
                
                # Adaptation level counts
                adaptation_level_counts = {}
                for level in AdaptationLevel:
                    count = base_query.filter(TestResultTable.adaptation_level == level.value).count()
                    adaptation_level_counts[level.value] = count
                
                # Failure reason counts
                failure_reason_counts = {}
                for reason in FailureReason:
                    count = base_query.filter(TestResultTable.failure_reason == reason.value).count()
                    failure_reason_counts[reason.value] = count
                
                # Average metrics
                avg_compilation_time = session.query(func.avg(TestResultTable.compilation_time_seconds))\
                    .filter(TestResultTable.compilation_time_seconds.isnot(None))\
                    .scalar()
                
                avg_execution_time = session.query(func.avg(TestResultTable.execution_time_seconds))\
                    .filter(TestResultTable.execution_time_seconds.isnot(None))\
                    .scalar()
                
                avg_memory_usage = session.query(func.avg(TestResultTable.memory_usage_mb))\
                    .filter(TestResultTable.memory_usage_mb.isnot(None))\
                    .scalar()
                
                # Success rate
                success_count = base_query.filter(TestResultTable.status == TestStatus.COMPLETED.value).count()
                success_rate = success_count / total_count if total_count > 0 else 0.0
                
                # Create statistics object
                return TestResultStatistics(
                    total_count=total_count,
                    status_counts=status_counts,
                    adaptation_level_counts=adaptation_level_counts,
                    failure_reason_counts=failure_reason_counts,
                    avg_compilation_time_seconds=avg_compilation_time,
                    avg_execution_time_seconds=avg_execution_time,
                    avg_memory_usage_mb=avg_memory_usage,
                    success_rate=success_rate
                )
                
        except Exception as e:
            logger.error(f"Error computing statistics: {str(e)}")
            return TestResultStatistics()

    def export_to_dataframe(self, filter_criteria: Optional[TestResultFilter] = None) -> pd.DataFrame:
        """
        Export test results to a pandas DataFrame.

        Args:
            filter_criteria: Optional criteria to filter results

        Returns:
            DataFrame containing test results
        """
        try:
            # Get filtered results
            records = self.filter_test_results(filter_criteria) if filter_criteria else self.filter_test_results(TestResultFilter())
            
            # Convert to DataFrame
            data = []
            for record in records:
                data.append({
                    "id": record.id,
                    "model_id": record.model_id,
                    "timestamp": record.timestamp,
                    "status": record.status.value,
                    "adaptation_level": record.adaptation_level.value if record.adaptation_level else None,
                    "failure_reason": record.failure_reason.value if record.failure_reason else None,
                    "compilation_time_seconds": record.compilation_time_seconds,
                    "execution_time_seconds": record.execution_time_seconds,
                    "memory_usage_mb": record.memory_usage_mb,
                    "output_matches_reference": record.output_matches_reference
                })
                
            return pd.DataFrame(data)
                
        except Exception as e:
            logger.error(f"Error exporting to DataFrame: {str(e)}")
            return pd.DataFrame()

    def _convert_to_test_record(self, db_record: TestResultTable) -> TestRecord:
        """
        Convert a database record to a TestRecord object.

        Args:
            db_record: Database record to convert

        Returns:
            TestRecord object
        """
        status = TestStatus(db_record.status)
        
        # Convert adaptation_level if it exists
        adaptation_level = None
        if db_record.adaptation_level:
            adaptation_level = AdaptationLevel(db_record.adaptation_level)
        
        # Convert failure_reason if it exists
        failure_reason = None
        if db_record.failure_reason:
            failure_reason = FailureReason(db_record.failure_reason)
        
        return TestRecord(
            id=db_record.id,
            model_id=db_record.model_id,
            timestamp=db_record.timestamp,
            status=status,
            adaptation_level=adaptation_level,
            failure_reason=failure_reason,
            error_message=db_record.error_message,
            stack_trace=db_record.stack_trace,
            compilation_time_seconds=db_record.compilation_time_seconds,
            execution_time_seconds=db_record.execution_time_seconds,
            memory_usage_mb=db_record.memory_usage_mb,
            adaptation_code=db_record.adaptation_code,
            output_matches_reference=db_record.output_matches_reference,
            metrics=db_record.metrics,
            tags=db_record.tags
        )
