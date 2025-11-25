"""
Prepare Segmentation Data for Training
======================================
Converts bounding box annotations to segmentation masks.
Handles multiple dataset formats.
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import json
from typing import List, Tuple, Dict

def create_mask_from_bboxes(
    image_shape: Tuple[int, int],
    rbc_boxes: List[Tuple[int, int, int, int]],
    wbc_boxes: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """
    Create segmentation mask from bounding boxes.
    
    Args:
        image_shape: (height, width)
        rbc_boxes: List of (x_min, y_min, x_max, y_max) for RBCs
        wbc_boxes: List of (x_min, y_min, x_max, y_max) for WBCs
    
    Returns:
        mask: (H, W, 3) RGB mask where:
            - Channel 0 (Red): Background = 255
            - Channel 1 (Green): RBC = 255
            - Channel 2 (Blue): WBC = 255
    """
    h, w = image_shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Set background
    mask[:, :, 0] = 255
    
    # Draw RBC regions (will overlap background)
    for x_min, y_min, x_max, y_max in rbc_boxes:
        # Create circular/elliptical mask for RBC (more accurate)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        radius_x = (x_max - x_min) // 2
        radius_y = (y_max - y_min) // 2
        
        cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 
                   0, 0, 360, (0, 255, 0), -1)
    
    # Draw WBC regions
    for x_min, y_min, x_max, y_max in wbc_boxes:
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        radius_x = (x_max - x_min) // 2
        radius_y = (y_max - y_min) // 2
        
        cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 
                   0, 0, 360, (0, 0, 255), -1)
    
    return mask


def parse_pascal_voc_xml(xml_path: str) -> Dict[str, List]:
    """
    Parse Pascal VOC format XML annotations.
    
    Returns:
        dict with 'rbc' and 'wbc' lists of bounding boxes
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    rbc_boxes = []
    wbc_boxes = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text.lower()
        bbox = obj.find('bndbox')
        
        x_min = int(bbox.find('xmin').text)
        y_min = int(bbox.find('ymin').text)
        x_max = int(bbox.find('xmax').text)
        y_max = int(bbox.find('ymax').text)
        
        # Classify based on name
        if 'rbc' in name or 'red' in name:
            rbc_boxes.append((x_min, y_min, x_max, y_max))
        elif 'wbc' in name or 'white' in name or 'leukocyte' in name:
            wbc_boxes.append((x_min, y_min, x_max, y_max))
    
    return {'rbc': rbc_boxes, 'wbc': wbc_boxes}


def process_bccd_dataset(
    bccd_root: str,
    output_dir: str
):
    """
    Process BCCD dataset to segmentation format.
    
    Directory structure expected:
    bccd_root/
        BCCD/
            JPEGImages/
            Annotations/
    
    Creates:
    output_dir/
        images/
        masks/
    """
    bccd_path = Path(bccd_root)
    output_path = Path(output_dir)
    
    images_dir = output_path / 'images'
    masks_dir = output_path / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    jpeg_dir = bccd_path / 'BCCD' / 'JPEGImages'
    ann_dir = bccd_path / 'BCCD' / 'Annotations'
    
    print(f"Processing BCCD dataset from {bccd_root}")
    print(f"Output to {output_dir}")
    
    processed = 0
    for img_path in jpeg_dir.glob('*.jpg'):
        xml_path = ann_dir / f"{img_path.stem}.xml"
        
        if not xml_path.exists():
            print(f"  Warning: No annotation for {img_path.name}")
            continue
        
        # Read image to get shape
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Error: Could not read {img_path.name}")
            continue
        
        h, w = image.shape[:2]
        
        # Parse annotations
        try:
            annotations = parse_pascal_voc_xml(str(xml_path))
        except Exception as e:
            print(f"  Error parsing {xml_path.name}: {e}")
            continue
        
        # Create mask
        mask = create_mask_from_bboxes(
            (h, w),
            annotations['rbc'],
            annotations['wbc']
        )
        
        # Save
        output_img_path = images_dir / img_path.name
        output_mask_path = masks_dir / img_path.name
        
        cv2.imwrite(str(output_img_path), image)
        cv2.imwrite(str(output_mask_path), mask)
        
        processed += 1
        if processed % 10 == 0:
            print(f"  Processed {processed} images...")
    
    print(f"\n✓ Processed {processed} images")
    print(f"  Images: {images_dir}")
    print(f"  Masks: {masks_dir}")


def create_augmented_dataset(
    source_dir: str,
    output_dir: str,
    augmentation_factor: int = 3
):
    """
    Create augmented dataset from source images and masks.
    Useful to increase dataset size from small datasets.
    
    Args:
        source_dir: Directory with images/ and masks/
        output_dir: Output directory
        augmentation_factor: How many augmented versions per image
    """
    import albumentations as A
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    out_images = output_path / 'images'
    out_masks = output_path / 'masks'
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    
    # Define augmentation pipeline
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.ElasticTransform(),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
        ], p=0.5),
        A.HueSaturationValue(p=0.3),
    ])
    
    images_dir = source_path / 'images'
    masks_dir = source_path / 'masks'
    
    print(f"Creating augmented dataset...")
    print(f"Augmentation factor: {augmentation_factor}")
    
    count = 0
    for img_path in images_dir.glob('*.png'):
        mask_path = masks_dir / img_path.name
        
        if not mask_path.exists():
            mask_path = masks_dir / f"{img_path.stem}.jpg"
        
        if not mask_path.exists():
            print(f"  Warning: No mask for {img_path.name}")
            continue
        
        # Read original
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Save original
        cv2.imwrite(str(out_images / img_path.name), 
                   cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_masks / img_path.name), 
                   cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
        count += 1
        
        # Create augmented versions
        for i in range(augmentation_factor):
            augmented = transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            # Save augmented
            aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            cv2.imwrite(str(out_images / aug_name), 
                       cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_masks / aug_name), 
                       cv2.cvtColor(aug_mask, cv2.COLOR_RGB2BGR))
            count += 1
        
        if count % 20 == 0:
            print(f"  Created {count} images...")
    
    print(f"\n✓ Created {count} images total")


def verify_dataset(data_dir: str):
    """Verify segmentation dataset is correctly formatted."""
    data_path = Path(data_dir)
    images_dir = data_path / 'images'
    masks_dir = data_path / 'masks'
    
    print(f"\nVerifying dataset: {data_dir}")
    print("="*60)
    
    if not images_dir.exists():
        print("❌ images/ directory not found")
        return False
    
    if not masks_dir.exists():
        print("❌ masks/ directory not found")
        return False
    
    image_files = list(images_dir.glob('*'))
    mask_files = list(masks_dir.glob('*'))
    
    print(f"✓ Found {len(image_files)} images")
    print(f"✓ Found {len(mask_files)} masks")
    
    # Check matching
    unmatched = []
    for img_path in image_files[:10]:  # Check first 10
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            unmatched.append(img_path.name)
    
    if unmatched:
        print(f"⚠️  Some images missing masks: {unmatched[:3]}...")
    else:
        print("✓ Images and masks are paired")
    
    # Check mask format
    sample_mask = cv2.imread(str(mask_files[0]))
    if sample_mask is None:
        print("❌ Could not read sample mask")
        return False
    
    print(f"✓ Mask shape: {sample_mask.shape}")
    print(f"✓ Mask dtype: {sample_mask.dtype}")
    
    # Check channels
    unique_vals = np.unique(sample_mask)
    print(f"✓ Unique values in mask: {unique_vals[:10]}")
    
    print("\n✓ Dataset verified!")
    return True


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare segmentation data')
    parser.add_argument('--mode', choices=['bccd', 'augment', 'verify'], 
                       required=True, help='Processing mode')
    parser.add_argument('--input', required=True, help='Input directory')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--aug-factor', type=int, default=3, 
                       help='Augmentation factor')
    
    args = parser.parse_args()
    
    if args.mode == 'bccd':
        process_bccd_dataset(args.input, args.output)
    elif args.mode == 'augment':
        create_augmented_dataset(args.input, args.output, args.aug_factor)
    elif args.mode == 'verify':
        verify_dataset(args.input)


if __name__ == "__main__":
    main()
