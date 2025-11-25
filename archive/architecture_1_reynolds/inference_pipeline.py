"""
End-to-End Inference Pipeline
==============================
Complete inference system that:
1. Takes raw microscopy images from microscope
2. Segments cells (RBC and WBC)
3. Classifies each cell type
4. Generates comprehensive diagnostic report
"""

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from dataclasses import dataclass
import json

from blood_cell_system import BloodCellAnalysisSystem


# ============================================================================
# DATA STRUCTURES FOR RESULTS
# ============================================================================

@dataclass
class CellDetection:
    """Single cell detection result"""
    cell_id: int
    cell_type: str  # 'RBC' or 'WBC'
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    disease_class: str
    confidence: float
    centroid: Tuple[int, int]


@dataclass
class DiagnosticReport:
    """Complete diagnostic report"""
    total_cells: int
    rbc_count: int
    wbc_count: int
    
    # RBC analysis
    healthy_rbc: int
    malaria_rbc: int
    sickle_rbc: int
    rbc_infection_rate: float
    
    # WBC analysis
    healthy_wbc: int
    leukemia_wbc: int
    wbc_cancer_rate: float
    
    # Detailed results
    cell_detections: List[CellDetection]
    uncertain_cases: List[Dict]
    
    # Image info
    image_path: str
    image_size: Tuple[int, int]
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'summary': {
                'total_cells': self.total_cells,
                'rbc_count': self.rbc_count,
                'wbc_count': self.wbc_count,
                'rbc_analysis': {
                    'healthy': self.healthy_rbc,
                    'malaria': self.malaria_rbc,
                    'sickle_cell': self.sickle_rbc,
                    'infection_rate': f"{self.rbc_infection_rate:.2%}"
                },
                'wbc_analysis': {
                    'healthy': self.healthy_wbc,
                    'leukemia': self.leukemia_wbc,
                    'cancer_rate': f"{self.wbc_cancer_rate:.2%}"
                }
            },
            'cell_detections': [
                {
                    'cell_id': c.cell_id,
                    'type': c.cell_type,
                    'disease': c.disease_class,
                    'confidence': float(c.confidence),
                    'bbox': c.bbox,
                    'centroid': c.centroid
                }
                for c in self.cell_detections
            ],
            'uncertain_cases': self.uncertain_cases,
            'image_info': {
                'path': self.image_path,
                'size': self.image_size
            }
        }
    
    def to_summary_text(self):
        """Generate human-readable summary"""
        summary = f"""
{'='*80}
BLOOD CELL ANALYSIS REPORT
{'='*80}

Image: {Path(self.image_path).name}
Image Size: {self.image_size[1]} x {self.image_size[0]} pixels

OVERALL STATISTICS:
-------------------
Total Cells Detected: {self.total_cells}
  - Red Blood Cells (RBC): {self.rbc_count}
  - White Blood Cells (WBC): {self.wbc_count}

RED BLOOD CELL ANALYSIS:
-----------------------
Healthy RBCs: {self.healthy_rbc} ({self.healthy_rbc/max(self.rbc_count,1)*100:.1f}%)
Malaria-infected RBCs: {self.malaria_rbc} ({self.malaria_rbc/max(self.rbc_count,1)*100:.1f}%)
Sickle Cell RBCs: {self.sickle_rbc} ({self.sickle_rbc/max(self.rbc_count,1)*100:.1f}%)
Overall RBC Infection Rate: {self.rbc_infection_rate:.2%}

WHITE BLOOD CELL ANALYSIS:
-------------------------
Healthy WBCs: {self.healthy_wbc} ({self.healthy_wbc/max(self.wbc_count,1)*100:.1f}%)
Leukemia WBCs: {self.leukemia_wbc} ({self.leukemia_wbc/max(self.wbc_count,1)*100:.1f}%)
Overall WBC Cancer Rate: {self.wbc_cancer_rate:.2%}
"""
        
        if self.uncertain_cases:
            summary += f"\nUNCERTAIN CASES: {len(self.uncertain_cases)}\n"
            summary += "-" * 40 + "\n"
            for case in self.uncertain_cases[:10]:  # Show first 10
                summary += f"  Cell #{case['cell_id']}: {case['predicted_class']} (confidence: {case['confidence']:.2%})\n"
            if len(self.uncertain_cases) > 10:
                summary += f"  ... and {len(self.uncertain_cases) - 10} more\n"
        
        summary += "\n" + "="*80 + "\n"
        return summary


# ============================================================================
# CELL EXTRACTION FROM SEGMENTATION
# ============================================================================

class CellExtractor:
    """Extracts individual cells from segmentation mask"""
    
    def __init__(
        self,
        min_area: int = 100,
        max_area: int = 10000,
        target_size: Tuple[int, int] = (130, 130)
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.target_size = target_size
    
    def extract_cells(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[List[np.ndarray], List[int], List[Tuple]]:
        """
        Extract individual cells from image using segmentation mask
        
        Args:
            image: Original RGB image (H, W, 3)
            mask: Segmentation mask (H, W, 3) with 3 channels for background/RBC/WBC
        
        Returns:
            cell_images: List of cropped and resized cell images
            cell_types: List of cell types (1=RBC, 2=WBC)
            cell_bboxes: List of bounding boxes (x_min, y_min, x_max, y_max)
        """
        # Convert mask to class labels
        mask_labels = np.argmax(mask, axis=-1)  # (H, W)
        
        cell_images = []
        cell_types = []
        cell_bboxes = []
        
        # Process RBCs (label = 1)
        rbc_mask = (mask_labels == 1).astype(np.uint8)
        rbc_cells, rbc_bboxes = self._extract_from_binary_mask(image, rbc_mask)
        cell_images.extend(rbc_cells)
        cell_types.extend([1] * len(rbc_cells))
        cell_bboxes.extend(rbc_bboxes)
        
        # Process WBCs (label = 2)
        wbc_mask = (mask_labels == 2).astype(np.uint8)
        wbc_cells, wbc_bboxes = self._extract_from_binary_mask(image, wbc_mask)
        cell_images.extend(wbc_cells)
        cell_types.extend([2] * len(wbc_cells))
        cell_bboxes.extend(wbc_bboxes)
        
        return cell_images, cell_types, cell_bboxes
    
    def _extract_from_binary_mask(
        self,
        image: np.ndarray,
        binary_mask: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple]]:
        """Extract cells from binary mask using connected components"""
        # Find connected components
        labeled_mask, num_labels = ndimage.label(binary_mask)
        
        cells = []
        bboxes = []
        
        for label_id in range(1, num_labels + 1):
            # Get pixels for this component
            component_mask = (labeled_mask == label_id)
            
            # Calculate area
            area = np.sum(component_mask)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding box
            coords = np.argwhere(component_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Add padding
            padding = 5
            h, w = image.shape[:2]
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(h, y_max + padding)
            x_max = min(w, x_max + padding)
            
            # Extract cell
            cell_crop = image[y_min:y_max, x_min:x_max]
            
            # Resize to target size
            cell_resized = cv2.resize(cell_crop, self.target_size)
            
            cells.append(cell_resized)
            bboxes.append((x_min, y_min, x_max, y_max))
        
        return cells, bboxes


# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

class InferencePipeline:
    """Complete end-to-end inference pipeline"""
    
    def __init__(
        self,
        system: BloodCellAnalysisSystem,
        confidence_threshold: float = 0.7,
        uncertain_margin: float = 0.15
    ):
        self.system = system
        self.confidence_threshold = confidence_threshold
        self.uncertain_margin = uncertain_margin
        self.cell_extractor = CellExtractor()
        
        # Class mappings
        self.rbc_classes = {0: 'healthy_RBC', 1: 'malaria_RBC', 2: 'sickle_RBC'}
        self.wbc_classes = {0: 'healthy_WBC', 1: 'leukemia_WBC'}
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image"""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32) / 255.0
        return img.numpy()
    
    def process_image(
        self,
        image_path: str,
        save_visualization: bool = True,
        output_dir: str = './results'
    ) -> DiagnosticReport:
        """
        Process a single microscopy image end-to-end
        
        Args:
            image_path: Path to input image
            save_visualization: Whether to save annotated image
            output_dir: Directory to save results
        
        Returns:
            DiagnosticReport with complete analysis
        """
        print(f"\nProcessing image: {image_path}")
        print("-" * 80)
        
        # 1. Load image
        print("Step 1/4: Loading image...")
        image = self.load_image(image_path)
        original_size = image.shape[:2]
        print(f"  Image size: {original_size[1]} x {original_size[0]}")
        
        # 2. Segment cells
        print("Step 2/4: Segmenting cells...")
        mask = self._segment_image(image)
        print(f"  Segmentation complete")
        
        # 3. Extract individual cells
        print("Step 3/4: Extracting individual cells...")
        cell_images, cell_types, cell_bboxes = self.cell_extractor.extract_cells(
            image, mask
        )
        rbc_count = sum(1 for t in cell_types if t == 1)
        wbc_count = sum(1 for t in cell_types if t == 2)
        print(f"  Detected {len(cell_images)} cells ({rbc_count} RBCs, {wbc_count} WBCs)")
        
        # 4. Classify cells
        print("Step 4/4: Classifying cells...")
        cell_detections, uncertain_cases = self._classify_cells(
            cell_images, cell_types, cell_bboxes
        )
        
        # 5. Generate report
        report = self._generate_report(
            cell_detections, uncertain_cases, image_path, original_size
        )
        
        # 6. Save visualization if requested
        if save_visualization:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self._save_visualization(image, cell_detections, report, image_path, output_dir)
        
        print("\n✓ Processing complete!")
        return report
    
    def _segment_image(self, image: np.ndarray) -> np.ndarray:
        """Run segmentation model"""
        # Resize to segmentation model input size
        h, w = self.system.seg_input_shape[:2]
        original_size = image.shape[:2]
        
        # Resize for model
        image_resized = tf.image.resize(image, (h, w))
        image_batch = tf.expand_dims(image_resized, 0)
        
        # Run segmentation
        mask = self.system.segmentation_model.predict(image_batch, verbose=0)[0]
        
        # Resize mask back to original size
        mask = tf.image.resize(mask, original_size)
        
        return mask.numpy()
    
    def _classify_cells(
        self,
        cell_images: List[np.ndarray],
        cell_types: List[int],
        cell_bboxes: List[Tuple]
    ) -> Tuple[List[CellDetection], List[Dict]]:
        """Classify all extracted cells"""
        cell_detections = []
        uncertain_cases = []
        
        # Separate RBCs and WBCs
        rbc_indices = [i for i, t in enumerate(cell_types) if t == 1]
        wbc_indices = [i for i, t in enumerate(cell_types) if t == 2]
        
        # Classify RBCs
        if rbc_indices:
            rbc_images = np.array([cell_images[i] for i in rbc_indices])
            
            if self.system.use_domain_adapt:
                rbc_preds, _ = self.system.rbc_classifier.predict(rbc_images, verbose=0)
            else:
                rbc_preds = self.system.rbc_classifier.predict(rbc_images, verbose=0)
            
            for idx, (pred, img_idx) in enumerate(zip(rbc_preds, rbc_indices)):
                class_id = np.argmax(pred)
                confidence = pred[class_id]
                
                detection = CellDetection(
                    cell_id=len(cell_detections),
                    cell_type='RBC',
                    bbox=cell_bboxes[img_idx],
                    disease_class=self.rbc_classes[class_id],
                    confidence=float(confidence),
                    centroid=self._get_centroid(cell_bboxes[img_idx])
                )
                cell_detections.append(detection)
                
                # Check if uncertain
                if confidence < self.confidence_threshold + self.uncertain_margin:
                    uncertain_cases.append({
                        'cell_id': detection.cell_id,
                        'cell_type': 'RBC',
                        'predicted_class': detection.disease_class,
                        'confidence': float(confidence)
                    })
        
        # Classify WBCs
        if wbc_indices:
            wbc_images = np.array([cell_images[i] for i in wbc_indices])
            
            if self.system.use_domain_adapt:
                wbc_preds, _ = self.system.wbc_classifier.predict(wbc_images, verbose=0)
            else:
                wbc_preds = self.system.wbc_classifier.predict(wbc_images, verbose=0)
            
            for idx, (pred, img_idx) in enumerate(zip(wbc_preds, wbc_indices)):
                class_id = np.argmax(pred)
                confidence = pred[class_id]
                
                detection = CellDetection(
                    cell_id=len(cell_detections),
                    cell_type='WBC',
                    bbox=cell_bboxes[img_idx],
                    disease_class=self.wbc_classes[class_id],
                    confidence=float(confidence),
                    centroid=self._get_centroid(cell_bboxes[img_idx])
                )
                cell_detections.append(detection)
                
                # Check if uncertain
                if confidence < self.confidence_threshold + self.uncertain_margin:
                    uncertain_cases.append({
                        'cell_id': detection.cell_id,
                        'cell_type': 'WBC',
                        'predicted_class': detection.disease_class,
                        'confidence': float(confidence)
                    })
        
        return cell_detections, uncertain_cases
    
    def _get_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Calculate centroid from bounding box"""
        x_min, y_min, x_max, y_max = bbox
        return ((x_min + x_max) // 2, (y_min + y_max) // 2)
    
    def _generate_report(
        self,
        cell_detections: List[CellDetection],
        uncertain_cases: List[Dict],
        image_path: str,
        image_size: Tuple[int, int]
    ) -> DiagnosticReport:
        """Generate comprehensive diagnostic report"""
        # Count cell types
        rbc_detections = [c for c in cell_detections if c.cell_type == 'RBC']
        wbc_detections = [c for c in cell_detections if c.cell_type == 'WBC']
        
        # Count RBC diseases
        healthy_rbc = sum(1 for c in rbc_detections if c.disease_class == 'healthy_RBC')
        malaria_rbc = sum(1 for c in rbc_detections if c.disease_class == 'malaria_RBC')
        sickle_rbc = sum(1 for c in rbc_detections if c.disease_class == 'sickle_RBC')
        
        # Count WBC diseases
        healthy_wbc = sum(1 for c in wbc_detections if c.disease_class == 'healthy_WBC')
        leukemia_wbc = sum(1 for c in wbc_detections if c.disease_class == 'leukemia_WBC')
        
        # Calculate rates
        rbc_infection_rate = (malaria_rbc + sickle_rbc) / max(len(rbc_detections), 1)
        wbc_cancer_rate = leukemia_wbc / max(len(wbc_detections), 1)
        
        return DiagnosticReport(
            total_cells=len(cell_detections),
            rbc_count=len(rbc_detections),
            wbc_count=len(wbc_detections),
            healthy_rbc=healthy_rbc,
            malaria_rbc=malaria_rbc,
            sickle_rbc=sickle_rbc,
            rbc_infection_rate=rbc_infection_rate,
            healthy_wbc=healthy_wbc,
            leukemia_wbc=leukemia_wbc,
            wbc_cancer_rate=wbc_cancer_rate,
            cell_detections=cell_detections,
            uncertain_cases=uncertain_cases,
            image_path=image_path,
            image_size=image_size
        )
    
    def _save_visualization(
        self,
        image: np.ndarray,
        cell_detections: List[CellDetection],
        report: DiagnosticReport,
        image_path: str,
        output_dir: str
    ):
        """Save annotated image with detection results"""
        # Convert to uint8 for OpenCV
        vis_image = (image * 255).astype(np.uint8).copy()
        
        # Color mapping for diseases
        colors = {
            'healthy_RBC': (0, 255, 0),      # Green
            'malaria_RBC': (0, 0, 255),      # Red
            'sickle_RBC': (255, 165, 0),     # Orange
            'healthy_WBC': (0, 255, 255),    # Yellow
            'leukemia_WBC': (255, 0, 255)    # Magenta
        }
        
        # Draw bounding boxes and labels
        for detection in cell_detections:
            x_min, y_min, x_max, y_max = detection.bbox
            color = colors[detection.disease_class]
            
            # Draw box
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw label
            label = f"{detection.disease_class} ({detection.confidence:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x_min, y_min - label_h - 5), (x_min + label_w, y_min), color, -1)
            cv2.putText(vis_image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save annotated image
        output_path = Path(output_dir) / f"{Path(image_path).stem}_annotated.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"\n  Annotated image saved to: {output_path}")
        
        # Save report as JSON
        report_path = Path(output_dir) / f"{Path(image_path).stem}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"  Report saved to: {report_path}")
        
        # Save text summary
        summary_path = Path(output_dir) / f"{Path(image_path).stem}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(report.to_summary_text())
        print(f"  Summary saved to: {summary_path}")


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def batch_process_images(
    image_dir: str,
    pipeline: InferencePipeline,
    output_dir: str = './results',
    extensions: List[str] = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
):
    """
    Process multiple images in batch
    
    Args:
        image_dir: Directory containing images
        pipeline: InferencePipeline instance
        output_dir: Output directory for results
        extensions: Allowed file extensions
    """
    # Find all images
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(image_dir).glob(f'*{ext}'))
        image_paths.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    print(f"\nFound {len(image_paths)} images to process")
    print("="*80)
    
    # Process each image
    reports = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"\nProcessing image {i}/{len(image_paths)}")
        try:
            report = pipeline.process_image(
                str(image_path),
                save_visualization=True,
                output_dir=output_dir
            )
            reports.append(report)
            print(report.to_summary_text())
        except Exception as e:
            print(f"ERROR processing {image_path}: {e}")
            continue
    
    # Save batch summary
    if reports:
        batch_summary = {
            'total_images': len(reports),
            'total_cells': sum(r.total_cells for r in reports),
            'total_rbcs': sum(r.rbc_count for r in reports),
            'total_wbcs': sum(r.wbc_count for r in reports),
            'malaria_cases': sum(r.malaria_rbc for r in reports),
            'sickle_cell_cases': sum(r.sickle_rbc for r in reports),
            'leukemia_cases': sum(r.leukemia_wbc for r in reports),
        }
        
        summary_path = Path(output_dir) / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print("BATCH PROCESSING COMPLETE!")
        print(f"{'='*80}")
        print(f"\nProcessed {batch_summary['total_images']} images")
        print(f"Total cells analyzed: {batch_summary['total_cells']}")
        print(f"  RBCs: {batch_summary['total_rbcs']}")
        print(f"  WBCs: {batch_summary['total_wbcs']}")
        print(f"\nDisease detections:")
        print(f"  Malaria: {batch_summary['malaria_cases']} cells")
        print(f"  Sickle Cell: {batch_summary['sickle_cell_cases']} cells")
        print(f"  Leukemia: {batch_summary['leukemia_cases']} cells")
        print(f"\nBatch summary saved to: {summary_path}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example inference script"""
    print("="*80)
    print("BLOOD CELL ANALYSIS - INFERENCE PIPELINE")
    print("="*80)
    
    # Load trained models
    print("\nLoading trained models...")
    system = BloodCellAnalysisSystem()
    system.load_models('./final_models')
    
    # Create inference pipeline
    pipeline = InferencePipeline(
        system=system,
        confidence_threshold=0.7,
        uncertain_margin=0.15
    )
    
    print("\nInference pipeline ready!")
    print("\nUsage examples:")
    print("  1. Single image:")
    print("     report = pipeline.process_image('path/to/image.png')")
    print("     print(report.to_summary_text())")
    print("\n  2. Batch processing:")
    print("     batch_process_images('path/to/images/', pipeline)")
    
    # Example (uncomment to run):
    # report = pipeline.process_image(
    #     'test_images/sample.png',
    #     save_visualization=True,
    #     output_dir='./results'
    # )
    # print(report.to_summary_text())


if __name__ == "__main__":
    main()
