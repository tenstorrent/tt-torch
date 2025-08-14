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
    JSON, DateTime, Text, desc, func, text
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
        
        # Enforce one-row-per-model: deduplicate and add a unique index on model_id
        try:
            with self.session_scope() as session:
                # Remove duplicates: keep the most recent by timestamp (then highest id)
                duplicates = (
                    session.query(TestResultTable.model_id, func.count('*').label('c'))
                    .group_by(TestResultTable.model_id)
                    .having(func.count('*') > 1)
                    .all()
                )
                for model_id, _ in duplicates:
                    rows = (
                        session.query(TestResultTable)
                        .filter(TestResultTable.model_id == model_id)
                        .order_by(desc(TestResultTable.timestamp), desc(TestResultTable.id))
                        .all()
                    )
                    for r in rows[1:]:
                        session.delete(r)
                # Create unique index (SQLite safe IF NOT EXISTS)
                session.execute(text('CREATE UNIQUE INDEX IF NOT EXISTS ux_test_results_model_id ON test_results (model_id)'))
        except Exception as e:
            logger.warning(f"Could not enforce unique model rows: {e}")

        logger.info(f"Initialized result tracking database at {db_url}")

    # Internal helpers
    def _apply_filters(self, query, filter_criteria: Optional[TestResultFilter]):
        if not filter_criteria:
            return query
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
        return query

    def _latest_per_model_query(self, session, filter_criteria: Optional[TestResultFilter]):
        # Apply timeframe and other filters first
        base = self._apply_filters(session.query(TestResultTable), filter_criteria)
        # Subquery to get latest timestamp per model within filtered set
        latest_ts_subq = (
            base.with_entities(
                TestResultTable.model_id,
                func.max(TestResultTable.timestamp).label('max_ts')
            )
            .group_by(TestResultTable.model_id)
            .subquery()
        )
        # Join back to rows
        latest_rows = (
            session.query(TestResultTable)
            .join(
                latest_ts_subq,
                (TestResultTable.model_id == latest_ts_subq.c.model_id) &
                (TestResultTable.timestamp == latest_ts_subq.c.max_ts)
            )
        )
        return latest_rows

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
        Insert or update a test result.

        - If `test_record.id` is set and exists in the DB, update that row in place
          (one row per model test run).
        - Otherwise insert a new row and return its ID.

        Args:
            test_record: TestRecord to insert/update

        Returns:
            ID of the stored record
        """
        try:
            with self.session_scope() as session:
                if getattr(test_record, 'id', None):
                    # Update existing row
                    db_record = session.query(TestResultTable).filter_by(id=test_record.id).first()
                    if db_record is not None:
                        # Update all fields to reflect latest run
                        db_record.model_id = test_record.model_id
                        db_record.timestamp = test_record.timestamp
                        db_record.status = test_record.status.value if hasattr(test_record.status, 'value') else str(test_record.status)
                        db_record.adaptation_level = (
                            test_record.adaptation_level.value
                            if (test_record.adaptation_level and hasattr(test_record.adaptation_level, 'value'))
                            else str(test_record.adaptation_level) if test_record.adaptation_level else None
                        )
                        db_record.failure_reason = (
                            test_record.failure_reason.value
                            if (test_record.failure_reason and hasattr(test_record.failure_reason, 'value'))
                            else str(test_record.failure_reason) if test_record.failure_reason else None
                        )
                        db_record.error_message = test_record.error_message
                        db_record.stack_trace = test_record.stack_trace
                        db_record.compilation_time_seconds = test_record.compilation_time_seconds
                        db_record.execution_time_seconds = test_record.execution_time_seconds
                        db_record.memory_usage_mb = test_record.memory_usage_mb
                        db_record.adaptation_code = test_record.adaptation_code
                        db_record.output_matches_reference = test_record.output_matches_reference
                        db_record.metrics = test_record.metrics
                        db_record.tags = test_record.tags
                        session.add(db_record)
                        session.flush()
                        logger.info(f"Updated test result for {test_record.model_id} (ID {db_record.id})")
                        return db_record.id
                    # If the id isn't found, fall through to model_id upsert/insert

                # If no id provided, try upsert by model_id (one row per model)
                existing = session.query(TestResultTable).filter_by(model_id=test_record.model_id).first()
                if existing is not None:
                    existing.timestamp = test_record.timestamp
                    existing.status = test_record.status.value if hasattr(test_record.status, 'value') else str(test_record.status)
                    existing.adaptation_level = (
                        test_record.adaptation_level.value
                        if (test_record.adaptation_level and hasattr(test_record.adaptation_level, 'value'))
                        else str(test_record.adaptation_level) if test_record.adaptation_level else None
                    )
                    existing.failure_reason = (
                        test_record.failure_reason.value
                        if (test_record.failure_reason and hasattr(test_record.failure_reason, 'value'))
                        else str(test_record.failure_reason) if test_record.failure_reason else None
                    )
                    existing.error_message = test_record.error_message
                    existing.stack_trace = test_record.stack_trace
                    existing.compilation_time_seconds = test_record.compilation_time_seconds
                    existing.execution_time_seconds = test_record.execution_time_seconds
                    existing.memory_usage_mb = test_record.memory_usage_mb
                    existing.adaptation_code = test_record.adaptation_code
                    existing.output_matches_reference = test_record.output_matches_reference
                    existing.metrics = test_record.metrics
                    existing.tags = test_record.tags
                    session.add(existing)
                    session.flush()
                    logger.info(f"Upserted test result by model_id for {test_record.model_id} (ID {existing.id})")
                    return existing.id

                # Insert new row
                db_record = TestResultTable(
                    model_id=test_record.model_id,
                    timestamp=test_record.timestamp,
                    status=test_record.status.value if hasattr(test_record.status, 'value') else str(test_record.status),
                    adaptation_level=test_record.adaptation_level.value if (test_record.adaptation_level and hasattr(test_record.adaptation_level, 'value')) else str(test_record.adaptation_level) if test_record.adaptation_level else None,
                    failure_reason=test_record.failure_reason.value if (test_record.failure_reason and hasattr(test_record.failure_reason, 'value')) else str(test_record.failure_reason) if test_record.failure_reason else None,
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
                logger.info(f"Inserted test result for {test_record.model_id} with ID {record_id}")
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
                query = self._apply_filters(session.query(TestResultTable), filter_criteria)
                
                # Sort by timestamp (newest first) BEFORE pagination (required by SQLAlchemy)
                query = query.order_by(desc(TestResultTable.timestamp))

                # Apply pagination AFTER order_by
                if filter_criteria.offset:
                    query = query.offset(filter_criteria.offset)
                if filter_criteria.limit:
                    query = query.limit(filter_criteria.limit)
                
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
                # Use ALL rows within filters (not just latest per model) so reruns fully affect stats
                filtered_q = self._apply_filters(session.query(TestResultTable), filter_criteria)
                rows = filtered_q.all()

                # Totals and aggregates computed in Python for clarity
                total_count = len(rows)

                # Status counts across all runs
                status_counts: Dict[str, int] = {}
                for status in TestStatus:
                    status_counts[status.value] = 0
                for r in rows:
                    status_str = str(r.status)
                    status_counts[status_str] = status_counts.get(status_str, 0) + 1

                # Adaptation level counts across all runs
                adaptation_level_counts: Dict[str, int] = {}
                for level in AdaptationLevel:
                    adaptation_level_counts[level.value] = 0
                for r in rows:
                    if r.adaptation_level:
                        level_str = str(r.adaptation_level)
                        adaptation_level_counts[level_str] = adaptation_level_counts.get(level_str, 0) + 1

                # Failure reason counts across all runs
                failure_reason_counts: Dict[str, int] = {}
                for reason in FailureReason:
                    failure_reason_counts[reason.value] = 0
                for r in rows:
                    if r.failure_reason:
                        reason_str = str(r.failure_reason)
                        failure_reason_counts[reason_str] = failure_reason_counts.get(reason_str, 0) + 1

                # Averages (ignore None)
                def _avg(values):
                    vals = [v for v in values if v is not None]
                    return sum(vals) / len(vals) if vals else None

                avg_compilation_time = _avg([r.compilation_time_seconds for r in rows])
                avg_execution_time = _avg([r.execution_time_seconds for r in rows])
                avg_memory_usage = _avg([r.memory_usage_mb for r in rows])

                # Success rate: fraction of runs with COMPLETED status
                success_count = sum(1 for r in rows if str(r.status) == TestStatus.COMPLETED.value)
                success_rate = success_count / total_count if total_count > 0 else 0.0

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
            # Use latest row per model (deduplicated)
            with self.session_scope() as session:
                latest_rows_q = self._latest_per_model_query(session, filter_criteria)
                records = [self._convert_to_test_record(r) for r in latest_rows_q.all()]
            
            # Convert to DataFrame
            data = []
            for record in records:
                data.append({
                    "id": record.id,
                    "model_id": record.model_id,
                    "timestamp": record.timestamp,
                    "status": record.status,  # These are already strings in the database
                    "adaptation_level": record.adaptation_level,
                    "failure_reason": record.failure_reason,
                    "compilation_time_seconds": record.compilation_time_seconds,
                    "execution_time_seconds": record.execution_time_seconds,
                    "memory_usage_mb": record.memory_usage_mb,
                    "output_matches_reference": record.output_matches_reference
                })
                
            return pd.DataFrame(data)
                
        except Exception as e:
            logger.error(f"Error exporting to DataFrame: {str(e)}")
            return pd.DataFrame()

    def delete_by_model_id(self, model_id: str) -> int:
        """Delete a test result row by exact model_id. Returns number of rows deleted."""
        try:
            with self.session_scope() as session:
                rows = (
                    session.query(TestResultTable)
                    .filter(TestResultTable.model_id == model_id)
                    .all()
                )
                deleted = 0
                for r in rows:
                    session.delete(r)
                    deleted += 1
                if deleted > 0:
                    logger.info(f"Deleted {deleted} row(s) for model_id={model_id}")
                return deleted
        except Exception as e:
            logger.error(f"Error deleting model_id {model_id}: {str(e)}")
            return 0

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
