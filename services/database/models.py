"""
Database Models - Architecture 2
================================
SQLAlchemy models for PostgreSQL database
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Patient(Base):
    """Patient information"""
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    reports = relationship("AnalysisReport", back_populates="patient")


class AnalysisReport(Base):
    """Analysis report"""
    __tablename__ = "analysis_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String, unique=True, index=True, nullable=False)
    patient_id = Column(Integer, nullable=False)
    
    # Image info
    image_path = Column(String, nullable=False)
    image_hash = Column(String, nullable=False)
    is_wsi = Column(Boolean, default=False)
    
    # Statistics
    total_cells = Column(Integer)
    rbc_count = Column(Integer)
    wbc_count = Column(Integer)
    rbc_abnormal_count = Column(Integer)
    rbc_abnormal_percentage = Column(Float)
    
    # Flags
    malaria_detected = Column(Boolean, default=False)
    sickle_cell_detected = Column(Boolean, default=False)
    leukemia_suspected = Column(Boolean, default=False)
    
    # Quality metrics
    image_quality_score = Column(Float)
    confidence_score = Column(Float)
    
    # Clinical output
    interpretation = Column(Text)
    recommendations = Column(JSON)
    
    # Full data
    statistics = Column(JSON)
    metadata = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float)  # seconds
    
    # Relationships
    patient = relationship("Patient", back_populates="reports")
    detections = relationship("CellDetection", back_populates="report")


class CellDetection(Base):
    """Individual cell detection"""
    __tablename__ = "cell_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, nullable=False)
    
    cell_type = Column(String, nullable=False)  # RBC or WBC
    classification = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    
    # Bounding box
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    
    # Relationships
    report = relationship("AnalysisReport", back_populates="detections")


class ModelCheckpoint(Base):
    """Model checkpoint tracking"""
    __tablename__ = "model_checkpoints"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    architecture = Column(String, nullable=False)
    
    checkpoint_path = Column(String, nullable=False)
    metrics = Column(JSON)
    
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database connection
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://postgres:changeme@localhost:5432/blood_cell_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database schema"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
