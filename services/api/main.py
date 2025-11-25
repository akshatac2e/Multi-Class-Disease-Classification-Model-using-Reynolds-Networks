"""
FastAPI Service Layer - Architecture 2
======================================
RESTful API for blood cell analysis system

Endpoints:
- POST /analyze - Analyze single image
- POST /analyze/batch - Batch analysis
- GET /report/{report_id} - Retrieve report
- GET /health - Health check
- GET /metrics - Prometheus metrics
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn
from pathlib import Path
import shutil
import uuid
from datetime import datetime
import logging

from inference_pipeline_v2 import (
    ProductionInferencePipeline,
    PatientInfo,
    DiagnosticReport
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Blood Cell Analysis API - Architecture 2",
    description="Production-grade blood cell analysis with HL7/FHIR integration",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (initialized on startup)
pipeline: Optional[ProductionInferencePipeline] = None

# Storage directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PatientInfoModel(BaseModel):
    """Patient information model"""
    patient_id: str
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    specimen_id: Optional[str] = None
    collection_date: Optional[str] = None


class AnalysisRequest(BaseModel):
    """Analysis request model"""
    patient_info: Optional[PatientInfoModel] = None
    output_formats: List[str] = Field(default=["json"], 
                                      description="Output formats: json, hl7, fhir")
    save_visualization: bool = True
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AnalysisResponse(BaseModel):
    """Analysis response model"""
    task_id: str
    status: str
    message: str
    report_id: Optional[str] = None
    processing_time: Optional[float] = None


class ReportSummary(BaseModel):
    """Report summary model"""
    report_id: str
    timestamp: str
    patient_id: str
    total_cells: int
    rbc_count: int
    wbc_count: int
    flags: Dict[str, bool]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    device: str
    models_loaded: bool


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    
    logger.info("Initializing production pipeline...")
    
    try:
        pipeline = ProductionInferencePipeline(
            checkpoint_dir="checkpoints",
            device="cuda"  # Configure from environment
        )
        logger.info("✓ Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "service": "Blood Cell Analysis API",
        "version": "2.0.0",
        "architecture": "Architecture 2 - Production",
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import torch
    
    return HealthResponse(
        status="healthy" if pipeline is not None else "unhealthy",
        version="2.0.0",
        device="cuda" if torch.cuda.is_available() else "cpu",
        models_loaded=pipeline is not None
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_id: str = "UNKNOWN",
    name: Optional[str] = None,
    output_formats: str = "json"
):
    """
    Analyze single blood cell image
    
    Parameters:
    - file: Image file (JPG, PNG, TIFF)
    - patient_id: Patient identifier
    - name: Patient name (optional)
    - output_formats: Comma-separated formats (json,hl7,fhir)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Parse output formats
    formats = [f.strip() for f in output_formats.split(",")]
    
    # Process image
    try:
        start_time = datetime.now()
        
        # Create patient info
        patient_info = PatientInfo(
            patient_id=patient_id,
            name=name
        )
        
        # Run analysis
        report = pipeline.process_single_image(
            image_path=str(upload_path),
            patient_info=patient_info,
            save_visualization=True
        )
        
        # Save reports
        pipeline.save_report(report, str(RESULTS_DIR), formats=formats)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Cleanup uploaded file in background
        background_tasks.add_task(lambda: upload_path.unlink(missing_ok=True))
        
        return AnalysisResponse(
            task_id=task_id,
            status="completed",
            message="Analysis completed successfully",
            report_id=report.report_id,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.get("/report/{report_id}", response_model=Dict)
async def get_report(report_id: str, format: str = "json"):
    """
    Retrieve analysis report
    
    Parameters:
    - report_id: Report identifier
    - format: Output format (json, hl7, fhir)
    """
    # Find report file
    if format == "json":
        report_path = RESULTS_DIR / f"{report_id}.json"
    elif format == "hl7":
        report_path = RESULTS_DIR / f"{report_id}.hl7"
    elif format == "fhir":
        report_path = RESULTS_DIR / f"{report_id}.fhir.json"
    else:
        raise HTTPException(status_code=400, detail=f"Invalid format: {format}")
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report not found: {report_id}")
    
    # Return file
    if format in ["json", "fhir"]:
        import json
        with open(report_path) as f:
            return json.load(f)
    else:  # HL7
        with open(report_path) as f:
            return {"hl7_message": f.read()}


@app.get("/report/{report_id}/visualization")
async def get_visualization(report_id: str):
    """Retrieve visualization image"""
    # Find visualization file in results
    vis_files = list(RESULTS_DIR.glob("*_analysis.jpg"))
    
    # Simple matching (in production, use proper database)
    for vis_file in vis_files:
        if report_id in vis_file.stem:
            return FileResponse(vis_file, media_type="image/jpeg")
    
    raise HTTPException(status_code=404, detail="Visualization not found")


@app.get("/reports", response_model=List[ReportSummary])
async def list_reports(limit: int = 100):
    """List recent reports"""
    reports = []
    
    for report_file in sorted(RESULTS_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            import json
            with open(report_file) as f:
                data = json.load(f)
            
            reports.append(ReportSummary(
                report_id=data['report_id'],
                timestamp=data['timestamp'],
                patient_id=data['patient']['patient_id'],
                total_cells=data['statistics']['total_cells_analyzed'],
                rbc_count=data['statistics']['rbc_count'],
                wbc_count=data['statistics']['wbc_count'],
                flags={
                    'malaria': data['statistics']['malaria_detected'],
                    'sickle_cell': data['statistics']['sickle_cell_detected'],
                    'leukemia': data['statistics']['leukemia_suspected']
                }
            ))
        except Exception as e:
            logger.warning(f"Failed to parse report {report_file}: {e}")
            continue
    
    return reports


@app.post("/analyze/batch")
async def analyze_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    patient_id_prefix: str = "BATCH"
):
    """
    Batch analysis endpoint
    
    Returns task IDs for tracking
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 files per batch")
    
    task_ids = []
    
    for idx, file in enumerate(files):
        task_id = str(uuid.uuid4())
        patient_id = f"{patient_id_prefix}_{idx+1:03d}"
        
        # Save file
        upload_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Schedule background processing
        background_tasks.add_task(
            process_batch_item,
            upload_path,
            patient_id,
            task_id
        )
        
        task_ids.append(task_id)
    
    return {
        "status": "accepted",
        "message": f"Batch of {len(files)} images queued for processing",
        "task_ids": task_ids
    }


async def process_batch_item(upload_path: Path, patient_id: str, task_id: str):
    """Background task for batch processing"""
    try:
        patient_info = PatientInfo(patient_id=patient_id)
        report = pipeline.process_single_image(
            image_path=str(upload_path),
            patient_info=patient_info
        )
        pipeline.save_report(report, str(RESULTS_DIR), formats=["json"])
        logger.info(f"Completed batch item: {task_id}")
    except Exception as e:
        logger.error(f"Batch item failed {task_id}: {e}")
    finally:
        upload_path.unlink(missing_ok=True)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    # TODO: Implement Prometheus metrics
    return {
        "requests_total": 0,
        "requests_success": 0,
        "requests_failed": 0,
        "average_processing_time": 0.0
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set True for development
        workers=1  # Increase for production
    )
