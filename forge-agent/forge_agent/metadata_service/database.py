"""
Database service for storing and querying model metadata.
"""
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import os

import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

from forge_agent.metadata_service.models import ModelMetadata, ModelSelectionCriteria, ModelFramework, ModelArchitecture

Base = declarative_base()


class ModelMetadataTable(Base):
    """SQLAlchemy model for model metadata."""
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String, unique=True, nullable=False, index=True)
    framework = Column(String, nullable=False, index=True)
    architecture = Column(String, nullable=False, index=True)
    downloads = Column(Integer, nullable=False, index=True)
    tags = Column(JSON, nullable=False)
    size_mb = Column(Float, nullable=True)
    last_modified = Column(String, nullable=True)
    is_compatible = Column(Boolean, nullable=True)
    raw_metadata = Column(JSON, nullable=False)


class MetadataDatabase:
    """Database service for model metadata."""

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize the metadata database.

        Args:
            db_url: SQLAlchemy database URL. If None, uses a default SQLite database.
        """
        if db_url is None:
            # Default to SQLite database in the package directory
            package_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(package_dir, "..", "..", "data", "model_metadata.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            db_url = f"sqlite:///{db_path}"
            
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

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

    def store_model_metadata(self, metadata: ModelMetadata) -> bool:
        """
        Store model metadata in the database.

        Args:
            metadata: ModelMetadata object

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_scope() as session:
                # Check if the model already exists
                existing = session.query(ModelMetadataTable).filter_by(model_id=metadata.model_id).first()
                
                if existing:
                    # Update existing record
                    existing.framework = metadata.framework.value
                    existing.architecture = metadata.architecture.value
                    existing.downloads = metadata.downloads
                    existing.tags = metadata.tags
                    existing.size_mb = metadata.size_mb
                    existing.last_modified = metadata.last_modified
                    existing.is_compatible = metadata.is_compatible
                    existing.raw_metadata = metadata.raw_metadata
                else:
                    # Create new record
                    model_data = ModelMetadataTable(
                        model_id=metadata.model_id,
                        framework=metadata.framework.value,
                        architecture=metadata.architecture.value,
                        downloads=metadata.downloads,
                        tags=metadata.tags,
                        size_mb=metadata.size_mb,
                        last_modified=metadata.last_modified,
                        is_compatible=metadata.is_compatible,
                        raw_metadata=metadata.raw_metadata
                    )
                    session.add(model_data)
                    
            return True
        except Exception as e:
            print(f"Error storing model metadata: {str(e)}")
            return False

    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata from the database.

        Args:
            model_id: Hugging Face model ID

        Returns:
            ModelMetadata object if found, None otherwise
        """
        try:
            with self.session_scope() as session:
                record = session.query(ModelMetadataTable).filter_by(model_id=model_id).first()
                
                if record:
                    return ModelMetadata(
                        model_id=record.model_id,
                        framework=ModelFramework(record.framework),
                        architecture=ModelArchitecture(record.architecture),
                        downloads=record.downloads,
                        tags=record.tags,
                        size_mb=record.size_mb,
                        last_modified=record.last_modified,
                        is_compatible=record.is_compatible,
                        raw_metadata=record.raw_metadata
                    )
                    
            return None
        except Exception as e:
            print(f"Error retrieving model metadata: {str(e)}")
            return None

    def filter_models(self, criteria: ModelSelectionCriteria) -> List[ModelMetadata]:
        """
        Filter models based on selection criteria.

        Args:
            criteria: ModelSelectionCriteria object

        Returns:
            List of ModelMetadata objects that match the criteria
        """
        try:
            with self.session_scope() as session:
                query = session.query(ModelMetadataTable)
                
                # Apply filters
                if criteria.min_downloads is not None:
                    query = query.filter(ModelMetadataTable.downloads >= criteria.min_downloads)
                
                if criteria.max_size_mb is not None:
                    query = query.filter(ModelMetadataTable.size_mb <= criteria.max_size_mb)
                
                if criteria.frameworks:
                    framework_values = [f.value for f in criteria.frameworks]
                    query = query.filter(ModelMetadataTable.framework.in_(framework_values))
                
                if criteria.architectures:
                    architecture_values = [a.value for a in criteria.architectures]
                    query = query.filter(ModelMetadataTable.architecture.in_(architecture_values))
                
                # Order by downloads (descending) and limit
                query = query.order_by(ModelMetadataTable.downloads.desc()).limit(criteria.limit)
                
                # Convert to ModelMetadata objects
                results = []
                for record in query.all():
                    results.append(ModelMetadata(
                        model_id=record.model_id,
                        framework=ModelFramework(record.framework),
                        architecture=ModelArchitecture(record.architecture),
                        downloads=record.downloads,
                        tags=record.tags,
                        size_mb=record.size_mb,
                        last_modified=record.last_modified,
                        is_compatible=record.is_compatible,
                        raw_metadata=record.raw_metadata
                    ))
                    
                return results
        except Exception as e:
            print(f"Error filtering models: {str(e)}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the models in the database.

        Returns:
            Dictionary with statistics
        """
        try:
            with self.session_scope() as session:
                # Total model count
                total_count = session.query(ModelMetadataTable).count()
                
                # Count by framework
                framework_counts = {}
                frameworks = session.query(ModelMetadataTable.framework, 
                                           text("COUNT(*) as count")).group_by(ModelMetadataTable.framework).all()
                for framework, count in frameworks:
                    framework_counts[framework] = count
                
                # Count by architecture
                architecture_counts = {}
                architectures = session.query(ModelMetadataTable.architecture, 
                                              text("COUNT(*) as count")).group_by(ModelMetadataTable.architecture).all()
                for architecture, count in architectures:
                    architecture_counts[architecture] = count
                
                # Count by compatibility
                compatible_count = session.query(ModelMetadataTable).filter_by(is_compatible=True).count()
                incompatible_count = session.query(ModelMetadataTable).filter_by(is_compatible=False).count()
                unknown_count = session.query(ModelMetadataTable).filter_by(is_compatible=None).count()
                
                return {
                    "total_count": total_count,
                    "framework_counts": framework_counts,
                    "architecture_counts": architecture_counts,
                    "compatible_count": compatible_count,
                    "incompatible_count": incompatible_count,
                    "unknown_count": unknown_count
                }
                
        except Exception as e:
            print(f"Error getting statistics: {str(e)}")
            return {
                "error": str(e)
            }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the database to a pandas DataFrame.

        Returns:
            DataFrame with model metadata
        """
        try:
            with self.session_scope() as session:
                # Query all records
                records = session.query(ModelMetadataTable).all()
                
                # Convert to dictionaries
                data = []
                for record in records:
                    data.append({
                        "model_id": record.model_id,
                        "framework": record.framework,
                        "architecture": record.architecture,
                        "downloads": record.downloads,
                        "size_mb": record.size_mb,
                        "is_compatible": record.is_compatible
                    })
                
                return pd.DataFrame(data)
        except Exception as e:
            print(f"Error converting to DataFrame: {str(e)}")
            return pd.DataFrame()
