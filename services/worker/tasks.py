"""
Celery Worker Tasks - Architecture 2
====================================
Async task processing for batch and long-running operations
"""

from celery import Celery
from pathlib import Path
import logging

# Initialize Celery
celery_app = Celery(
    'blood_cell_worker',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes
    task_soft_time_limit=540  # 9 minutes
)

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='tasks.analyze_image')
def analyze_image_task(self, image_path: str, patient_id: str, output_formats: list):
    """
    Celery task for image analysis
    """
    from inference_pipeline_v2 import ProductionInferencePipeline, PatientInfo
    
    try:
        # Update state
        self.update_state(state='PROCESSING', meta={'progress': 0})
        
        # Initialize pipeline
        pipeline = ProductionInferencePipeline(
            checkpoint_dir="checkpoints",
            device="cuda"
        )
        
        self.update_state(state='PROCESSING', meta={'progress': 20})
        
        # Process image
        patient_info = PatientInfo(patient_id=patient_id)
        report = pipeline.process_single_image(
            image_path=image_path,
            patient_info=patient_info
        )
        
        self.update_state(state='PROCESSING', meta={'progress': 80})
        
        # Save report
        pipeline.save_report(report, "results", formats=output_formats)
        
        return {
            'status': 'completed',
            'report_id': report.report_id,
            'statistics': report.statistics.to_dict()
        }
    
    except Exception as e:
        logger.error(f"Task failed: {e}")
        raise


@celery_app.task(name='tasks.batch_analysis')
def batch_analysis_task(image_paths: list, patient_ids: list):
    """
    Batch analysis task
    """
    results = []
    
    for image_path, patient_id in zip(image_paths, patient_ids):
        result = analyze_image_task.delay(image_path, patient_id, ['json'])
        results.append(result.id)
    
    return {'task_ids': results}


if __name__ == '__main__':
    celery_app.start()
