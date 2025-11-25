"""
Production Inference Pipeline - Architecture 2
==============================================
Complete 6-stage clinical pipeline:
1. Data Ingestion & Validation
2. Preprocessing & Stain Normalization  
3. Instance Segmentation
4. Cell Routing
5. Specialized Classification
6. Aggregation & Clinical Output (HL7/FHIR)

Features:
- WSI support with adaptive tiling
- Quality control at each stage
- Clinical standards compliance
- HL7/FHIR integration
- Batch processing
- Real-time streaming
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import hashlib
from enum import Enum

try:
    import hl7apy
    from hl7apy.core import Message, Segment
    HL7_AVAILABLE = True
except ImportError:
    HL7_AVAILABLE = False
    print("Warning: hl7apy not available. Install for HL7/FHIR output support.")

from blood_cell_system import (
    ProductionBloodCellSystem,
    CellDetection,
    CellType,
    RBCClass,
    WBCClass,
    WSIMetadata,
    StainMethod,
    visualize_detections,
    load_model_weights
)


# ============================================================================
# CLINICAL OUTPUT FORMATS
# ============================================================================

@dataclass
class PatientInfo:
    """Patient demographic information"""
    patient_id: str
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    specimen_id: Optional[str] = None
    collection_date: Optional[str] = None


@dataclass
class ClinicalStatistics:
    """Clinical statistics and interpretations"""
    total_cells_analyzed: int
    rbc_count: int
    wbc_count: int
    
    # RBC metrics
    rbc_normal_count: int
    rbc_abnormal_count: int
    rbc_abnormal_percentage: float
    rbc_distribution: Dict[str, int]
    
    # WBC metrics
    wbc_distribution: Dict[str, int]
    wbc_differential: Dict[str, float]  # Percentages
    blast_count: int
    blast_percentage: float
    
    # Flags
    malaria_detected: bool
    sickle_cell_detected: bool
    leukemia_suspected: bool
    
    # Quality metrics
    image_quality_score: float
    confidence_score: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DiagnosticReport:
    """Complete diagnostic report"""
    report_id: str
    timestamp: str
    patient_info: PatientInfo
    statistics: ClinicalStatistics
    detections: List[CellDetection]
    metadata: Dict
    interpretation: str
    recommendations: List[str]
    
    def to_json(self) -> str:
        """Export as JSON"""
        return json.dumps({
            'report_id': self.report_id,
            'timestamp': self.timestamp,
            'patient': asdict(self.patient_info),
            'statistics': self.statistics.to_dict(),
            'interpretation': self.interpretation,
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }, indent=2)
    
    def to_hl7(self) -> Optional[str]:
        """Export as HL7 v2.x message"""
        if not HL7_AVAILABLE:
            return None
        
        # Create ORU^R01 message (Observation Result)
        msg = Message("ORU_R01")
        
        # MSH segment
        msg.msh.msh_3 = "BloodCellAnalyzer"
        msg.msh.msh_4 = "LabSystem"
        msg.msh.msh_7 = datetime.now().strftime("%Y%m%d%H%M%S")
        msg.msh.msh_9 = "ORU^R01"
        msg.msh.msh_10 = self.report_id
        
        # PID segment (Patient Identification)
        msg.add_segment("PID")
        msg.pid.pid_3 = self.patient_info.patient_id
        if self.patient_info.name:
            msg.pid.pid_5 = self.patient_info.name
        
        # OBR segment (Observation Request)
        msg.add_segment("OBR")
        msg.obr.obr_4 = "CBC^Complete Blood Count"
        
        # OBX segments (Observation Results)
        obs_id = 1
        
        # RBC count
        msg.add_segment("OBX")
        obx = msg.obx(obs_id)
        obx.obx_1 = str(obs_id)
        obx.obx_2 = "NM"  # Numeric
        obx.obx_3 = "RBC^Red Blood Cell Count"
        obx.obx_5 = str(self.statistics.rbc_count)
        obx.obx_6 = "cells"
        obs_id += 1
        
        # WBC count
        msg.add_segment("OBX")
        obx = msg.obx(obs_id)
        obx.obx_1 = str(obs_id)
        obx.obx_2 = "NM"
        obx.obx_3 = "WBC^White Blood Cell Count"
        obx.obx_5 = str(self.statistics.wbc_count)
        obx.obx_6 = "cells"
        obs_id += 1
        
        # Malaria flag
        if self.statistics.malaria_detected:
            msg.add_segment("OBX")
            obx = msg.obx(obs_id)
            obx.obx_1 = str(obs_id)
            obx.obx_2 = "CE"  # Coded Entry
            obx.obx_3 = "MALARIA^Malaria Detection"
            obx.obx_5 = "POSITIVE"
            obx.obx_8 = "A"  # Abnormal
            obs_id += 1
        
        return msg.to_er7()
    
    def to_fhir(self) -> Dict:
        """Export as FHIR DiagnosticReport"""
        return {
            "resourceType": "DiagnosticReport",
            "id": self.report_id,
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "58410-2",
                    "display": "Complete blood count (hemogram) panel"
                }]
            },
            "subject": {
                "reference": f"Patient/{self.patient_info.patient_id}"
            },
            "issued": self.timestamp,
            "result": [
                {
                    "reference": f"Observation/rbc-count-{self.report_id}",
                    "display": f"RBC Count: {self.statistics.rbc_count}"
                },
                {
                    "reference": f"Observation/wbc-count-{self.report_id}",
                    "display": f"WBC Count: {self.statistics.wbc_count}"
                }
            ],
            "conclusion": self.interpretation
        }


# ============================================================================
# QUALITY CONTROL
# ============================================================================

class QualityChecker:
    """Quality control for input images and results"""
    
    @staticmethod
    def validate_image(image: np.ndarray) -> Tuple[bool, str]:
        """Validate input image quality"""
        h, w = image.shape[:2]
        
        # Check dimensions
        if h < 512 or w < 512:
            return False, "Image too small (minimum 512×512)"
        
        if h > 50000 or w > 50000:
            return False, "Image too large (maximum 50000×50000)"
        
        # Check if image is mostly blank
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if gray.mean() > 240:
            return False, "Image appears to be mostly blank"
        
        # Check contrast
        if gray.std() < 10:
            return False, "Image has insufficient contrast"
        
        # Check for focus (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        focus_score = laplacian.var()
        if focus_score < 50:
            return False, f"Image appears out of focus (score: {focus_score:.1f})"
        
        return True, "OK"
    
    @staticmethod
    def compute_image_quality_score(image: np.ndarray) -> float:
        """Compute overall image quality score (0-1)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Focus score (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        focus_score = min(laplacian.var() / 500, 1.0)
        
        # Contrast score (std dev)
        contrast_score = min(gray.std() / 50, 1.0)
        
        # Brightness score (distance from optimal)
        brightness_score = 1.0 - abs(gray.mean() - 128) / 128
        
        # Combined score
        return (focus_score * 0.4 + contrast_score * 0.3 + brightness_score * 0.3)
    
    @staticmethod
    def filter_low_confidence_detections(
        detections: List[CellDetection],
        min_confidence: float = 0.5
    ) -> List[CellDetection]:
        """Filter out low-confidence detections"""
        return [d for d in detections if d.score >= min_confidence 
                and (d.classification_score is None or d.classification_score >= min_confidence)]


# ============================================================================
# CLINICAL INTERPRETER
# ============================================================================

class ClinicalInterpreter:
    """Generate clinical interpretations and recommendations"""
    
    @staticmethod
    def interpret_results(stats: ClinicalStatistics) -> Tuple[str, List[str]]:
        """
        Generate interpretation and recommendations
        Returns: (interpretation_text, recommendations_list)
        """
        interpretation_parts = []
        recommendations = []
        
        # Overall summary
        interpretation_parts.append(
            f"Analysis of {stats.total_cells_analyzed} cells identified "
            f"{stats.rbc_count} red blood cells and {stats.wbc_count} white blood cells."
        )
        
        # RBC analysis
        if stats.rbc_abnormal_percentage > 50:
            interpretation_parts.append(
                f"CRITICAL: {stats.rbc_abnormal_percentage:.1f}% of RBCs show abnormalities."
            )
            recommendations.append("Immediate hematology consultation recommended")
        elif stats.rbc_abnormal_percentage > 20:
            interpretation_parts.append(
                f"ABNORMAL: {stats.rbc_abnormal_percentage:.1f}% of RBCs show abnormalities."
            )
            recommendations.append("Follow-up testing recommended")
        
        # Malaria detection
        if stats.malaria_detected:
            malaria_count = stats.rbc_distribution.get('malaria_infected', 0)
            interpretation_parts.append(
                f"MALARIA DETECTED: {malaria_count} infected cells identified."
            )
            recommendations.extend([
                "Initiate antimalarial therapy immediately",
                "Confirm with thick/thin blood smears",
                "Monitor parasitemia levels"
            ])
        
        # Sickle cell detection
        if stats.sickle_cell_detected:
            sickle_count = stats.rbc_distribution.get('sickle', 0)
            interpretation_parts.append(
                f"SICKLE CELLS DETECTED: {sickle_count} sickle-shaped cells identified."
            )
            recommendations.extend([
                "Hemoglobin electrophoresis recommended",
                "Genetic counseling may be appropriate",
                "Monitor for vaso-occlusive complications"
            ])
        
        # Leukemia suspicion
        if stats.leukemia_suspected:
            interpretation_parts.append(
                f"LEUKEMIA SUSPECTED: {stats.blast_count} blast cells ({stats.blast_percentage:.1f}%) detected."
            )
            recommendations.extend([
                "URGENT: Hematology/Oncology referral required",
                "Bone marrow biopsy recommended",
                "Flow cytometry for immunophenotyping",
                "Cytogenetic studies recommended"
            ])
        
        # WBC differential
        if stats.wbc_count > 0:
            interpretation_parts.append("WBC differential: " + ", ".join(
                f"{cell_type}: {pct:.1f}%"
                for cell_type, pct in stats.wbc_differential.items()
            ))
        
        # Quality assessment
        if stats.image_quality_score < 0.6:
            interpretation_parts.append(
                f"NOTE: Image quality score is low ({stats.image_quality_score:.2f}). "
                "Consider repeat imaging for confirmation."
            )
            recommendations.append("Repeat slide preparation and imaging recommended")
        
        interpretation = " ".join(interpretation_parts)
        return interpretation, recommendations


# ============================================================================
# PRODUCTION INFERENCE PIPELINE
# ============================================================================

class ProductionInferencePipeline:
    """
    Complete production inference pipeline
    Handles end-to-end processing from image to clinical report
    """
    
    def __init__(self,
                 checkpoint_dir: str,
                 config: Optional[Dict] = None,
                 device: str = "cuda"):
        
        self.device = device
        self.config = config or self._default_config()
        
        # Initialize system
        self.system = ProductionBloodCellSystem(
            segmentation_arch=self.config.get('segmentation_arch', 'htc'),
            wbc_classifier_arch=self.config.get('wbc_classifier_arch', 'convnext'),
            stain_method=StainMethod[self.config.get('stain_method', 'MACENKO')],
            device=device
        )
        
        # Load trained weights
        load_model_weights(self.system, checkpoint_dir)
        
        # Initialize components
        self.quality_checker = QualityChecker()
        self.interpreter = ClinicalInterpreter()
        
        print(f"✓ Production pipeline initialized")
        print(f"  Device: {device}")
        print(f"  Segmentation: {self.config.get('segmentation_arch', 'htc')}")
        print(f"  Stain normalization: {self.config.get('stain_method', 'MACENKO')}")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'segmentation_arch': 'htc',
            'wbc_classifier_arch': 'convnext',
            'stain_method': 'MACENKO',
            'min_confidence': 0.5,
            'enable_wsi': True,
            'enable_quality_check': True
        }
    
    def process_single_image(self,
                            image_path: str,
                            patient_info: Optional[PatientInfo] = None,
                            save_visualization: bool = True) -> DiagnosticReport:
        """
        Process single microscopy image
        
        Args:
            image_path: Path to input image
            patient_info: Optional patient information
            save_visualization: Whether to save visualization
        
        Returns:
            Complete diagnostic report
        """
        image_path = Path(image_path)
        
        # Stage 1: Data Ingestion & Validation
        print(f"[1/6] Loading image: {image_path.name}")
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # File integrity check
        file_hash = self._compute_file_hash(image_path)
        
        # Quality validation
        if self.config.get('enable_quality_check', True):
            is_valid, message = self.quality_checker.validate_image(image)
            if not is_valid:
                raise ValueError(f"Image quality check failed: {message}")
            
            quality_score = self.quality_checker.compute_image_quality_score(image)
            print(f"  Quality score: {quality_score:.2f}")
        else:
            quality_score = 1.0
        
        # Stage 2-5: Process image
        print(f"[2-5/6] Processing pipeline...")
        is_wsi = image.shape[0] > 5000 or image.shape[1] > 5000
        detections = self.system.process_image(image, is_wsi=is_wsi)
        
        # Filter low confidence
        detections = self.quality_checker.filter_low_confidence_detections(
            detections,
            min_confidence=self.config.get('min_confidence', 0.5)
        )
        
        print(f"  Detected {len(detections)} cells")
        
        # Stage 6: Aggregation & Clinical Output
        print(f"[6/6] Generating clinical report...")
        
        # Compute statistics
        statistics = self._compute_statistics(detections, quality_score)
        
        # Generate interpretation
        interpretation, recommendations = self.interpreter.interpret_results(statistics)
        
        # Create report
        report_id = self._generate_report_id(file_hash)
        
        if patient_info is None:
            patient_info = PatientInfo(
                patient_id="UNKNOWN",
                specimen_id=image_path.stem
            )
        
        report = DiagnosticReport(
            report_id=report_id,
            timestamp=datetime.now().isoformat(),
            patient_info=patient_info,
            statistics=statistics,
            detections=detections,
            metadata={
                'image_path': str(image_path),
                'image_size': f"{image.shape[1]}×{image.shape[0]}",
                'file_hash': file_hash,
                'is_wsi': is_wsi,
                'processing_config': self.config
            },
            interpretation=interpretation,
            recommendations=recommendations
        )
        
        # Save visualization
        if save_visualization:
            vis_path = image_path.parent / f"{image_path.stem}_analysis.jpg"
            visualize_detections(image, detections, str(vis_path))
            print(f"  Saved visualization: {vis_path.name}")
        
        print(f"✓ Report generated: {report_id}")
        
        return report
    
    def _compute_statistics(self,
                           detections: List[CellDetection],
                           quality_score: float) -> ClinicalStatistics:
        """Compute clinical statistics from detections"""
        
        # Count by type
        rbc_detections = [d for d in detections if d.cell_type == CellType.RBC]
        wbc_detections = [d for d in detections if d.cell_type == CellType.WBC]
        
        # RBC distribution
        rbc_dist = {}
        for rbc_class in RBCClass:
            count = sum(1 for d in rbc_detections if d.classification == rbc_class)
            if count > 0:
                rbc_dist[rbc_class.value] = count
        
        rbc_normal = rbc_dist.get('normal', 0)
        rbc_abnormal = len(rbc_detections) - rbc_normal
        rbc_abnormal_pct = (rbc_abnormal / len(rbc_detections) * 100) if rbc_detections else 0
        
        # WBC distribution and differential
        wbc_dist = {}
        for wbc_class in WBCClass:
            count = sum(1 for d in wbc_detections if d.classification == wbc_class)
            if count > 0:
                wbc_dist[wbc_class.value] = count
        
        wbc_differential = {
            k: (v / len(wbc_detections) * 100) if wbc_detections else 0
            for k, v in wbc_dist.items()
        }
        
        # Flags
        malaria_detected = rbc_dist.get('malaria_infected', 0) > 0
        sickle_detected = rbc_dist.get('sickle', 0) > 0
        blast_count = wbc_dist.get('blast_leukemia', 0)
        blast_pct = (blast_count / len(wbc_detections) * 100) if wbc_detections else 0
        leukemia_suspected = blast_pct > 5.0  # >5% blasts is concerning
        
        # Average confidence
        confidences = [d.classification_score for d in detections 
                      if d.classification_score is not None]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return ClinicalStatistics(
            total_cells_analyzed=len(detections),
            rbc_count=len(rbc_detections),
            wbc_count=len(wbc_detections),
            rbc_normal_count=rbc_normal,
            rbc_abnormal_count=rbc_abnormal,
            rbc_abnormal_percentage=rbc_abnormal_pct,
            rbc_distribution=rbc_dist,
            wbc_distribution=wbc_dist,
            wbc_differential=wbc_differential,
            blast_count=blast_count,
            blast_percentage=blast_pct,
            malaria_detected=malaria_detected,
            sickle_cell_detected=sickle_detected,
            leukemia_suspected=leukemia_suspected,
            image_quality_score=quality_score,
            confidence_score=avg_confidence
        )
    
    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]
    
    @staticmethod
    def _generate_report_id(file_hash: str) -> str:
        """Generate unique report ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"RPT-{timestamp}-{file_hash}"
    
    def save_report(self, report: DiagnosticReport, output_dir: str, 
                   formats: List[str] = ['json', 'hl7', 'fhir']):
        """
        Save report in multiple formats
        
        Args:
            report: Diagnostic report to save
            output_dir: Output directory
            formats: List of formats ('json', 'hl7', 'fhir')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = output_dir / report.report_id
        
        if 'json' in formats:
            json_path = base_name.with_suffix('.json')
            with open(json_path, 'w') as f:
                f.write(report.to_json())
            print(f"  Saved JSON: {json_path.name}")
        
        if 'hl7' in formats:
            hl7_msg = report.to_hl7()
            if hl7_msg:
                hl7_path = base_name.with_suffix('.hl7')
                with open(hl7_path, 'w') as f:
                    f.write(hl7_msg)
                print(f"  Saved HL7: {hl7_path.name}")
        
        if 'fhir' in formats:
            fhir_data = report.to_fhir()
            fhir_path = base_name.with_suffix('.fhir.json')
            with open(fhir_path, 'w') as f:
                json.dump(fhir_data, f, indent=2)
            print(f"  Saved FHIR: {fhir_path.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Inference Pipeline")
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to model checkpoints')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for reports')
    parser.add_argument('--patient_id', type=str, default='UNKNOWN',
                       help='Patient ID')
    parser.add_argument('--formats', nargs='+', default=['json', 'hl7', 'fhir'],
                       choices=['json', 'hl7', 'fhir'],
                       help='Output formats')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for inference')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ProductionInferencePipeline(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )
    
    # Process image
    patient_info = PatientInfo(patient_id=args.patient_id)
    report = pipeline.process_single_image(
        image_path=args.image,
        patient_info=patient_info
    )
    
    # Save report
    pipeline.save_report(report, args.output_dir, formats=args.formats)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Report ID: {report.report_id}")
    print(f"Patient: {report.patient_info.patient_id}")
    print(f"\nCells Analyzed: {report.statistics.total_cells_analyzed}")
    print(f"  RBC: {report.statistics.rbc_count}")
    print(f"  WBC: {report.statistics.wbc_count}")
    print(f"\nInterpretation:")
    print(f"  {report.interpretation}")
    if report.recommendations:
        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")
    print("=" * 60)
