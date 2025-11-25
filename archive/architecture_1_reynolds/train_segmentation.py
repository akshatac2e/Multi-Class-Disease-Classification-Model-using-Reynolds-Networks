"""
Train Segmentation Model
========================
Complete training script for the segmentation model.
Handles multiple staining methods through domain adaptation.
"""

import tensorflow as tf
from blood_cell_system import BloodCellAnalysisSystem
from training_pipeline import TrainingPipeline
import argparse
from pathlib import Path
import yaml

def train_segmentation(
    data_dir: str,
    config_preset: str = 'balanced',
    epochs: int = 50,
    batch_size: int = 8,
    resume_from: str = None
):
    """
    Train segmentation model.
    
    Args:
        data_dir: Directory with images/ and masks/ subdirectories
        config_preset: 'fast', 'balanced', 'accurate', or 'research'
        epochs: Number of training epochs
        batch_size: Batch size (8 recommended, reduce if OOM)
        resume_from: Path to checkpoint to resume from
    """
    print("="*80)
    print("TRAINING SEGMENTATION MODEL")
    print("="*80)
    
    # Load config preset
    config_file = Path('config/training_presets.yaml')
    if config_file.exists():
        with open(config_file) as f:
            presets = yaml.safe_load(f)
            preset_config = presets.get(config_preset, presets['balanced'])
    else:
        preset_config = {'d_reduced': 64, 'use_domain_adapt': True}
    
    print(f"\nUsing preset: {config_preset}")
    print(f"Configuration: {preset_config}")
    
    # Build system
    print("\n[1/4] Building segmentation model...")
    system = BloodCellAnalysisSystem(
        seg_input_shape=(512, 512, 3),
        d_reduced=preset_config.get('d_reduced', 64),
        use_domain_adapt=preset_config.get('use_domain_adapt', True)
    )
    
    system.build_all_models()  # Only builds, we'll train segmentation
    
    # Compile with custom settings for segmentation
    learning_rate = preset_config.get('learning_rate', 1e-4)
    system.segmentation_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.MeanIoU(num_classes=3, name='mean_iou')
        ]
    )
    
    print(f"✓ Model compiled with lr={learning_rate}")
    
    # Load checkpoint if resuming
    if resume_from:
        print(f"\n[2/4] Loading checkpoint: {resume_from}")
        system.segmentation_model.load_weights(resume_from)
        print("✓ Checkpoint loaded")
    else:
        print("\n[2/4] Training from scratch")
    
    # Create training pipeline
    print("\n[3/4] Preparing data...")
    config = {
        'dataset_type': 'segmentation',
        'use_domain_adaptation': preset_config.get('use_domain_adapt', True),
        'd_reduced': preset_config.get('d_reduced', 64)
    }
    pipeline = TrainingPipeline(system, config)
    
    # Train
    print(f"\n[4/4] Training for {epochs} epochs...")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Expected time: ~{epochs * 2.5:.1f} minutes\n")
    
    try:
        history = pipeline.train_segmentation_model(
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            save_best=True
        )
        
        # Save final model
        print("\nSaving final model...")
        system.segmentation_model.save('models/segmentation_model_final.keras')
        
        # Print summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_iou = history.history['mean_iou'][-1]
        final_val_iou = history.history['val_mean_iou'][-1]
        
        print(f"\nFinal Metrics:")
        print(f"  Training Accuracy: {final_acc:.4f}")
        print(f"  Validation Accuracy: {final_val_acc:.4f}")
        print(f"  Training IoU: {final_iou:.4f}")
        print(f"  Validation IoU: {final_val_iou:.4f}")
        
        print(f"\nModels saved:")
        print(f"  Best: checkpoints/segmentation_model_best.keras")
        print(f"  Final: models/segmentation_model_final.keras")
        
        if final_val_iou > 0.85:
            print("\n✓ Excellent performance! (IoU > 0.85)")
        elif final_val_iou > 0.75:
            print("\n✓ Good performance! (IoU > 0.75)")
        else:
            print("\n⚠️  Consider training longer or checking data quality")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


def test_segmentation(
    model_path: str,
    test_image: str,
    output_dir: str = './test_results'
):
    """
    Test trained segmentation model on a single image.
    
    Args:
        model_path: Path to trained model
        test_image: Path to test image
        output_dir: Directory to save results
    """
    import cv2
    import numpy as np
    from pathlib import Path
    
    print("\n" + "="*80)
    print("TESTING SEGMENTATION MODEL")
    print("="*80)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    system = BloodCellAnalysisSystem()
    system.segmentation_model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'StainNormalizationLayer': system.segmentation_model.layers[1].__class__,
        }
    )
    print("✓ Model loaded")
    
    # Load image
    print(f"\nLoading test image: {test_image}")
    img = tf.io.read_file(test_image)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img_original = img.numpy()
    
    # Resize for model
    img_resized = tf.image.resize(img, (512, 512)) / 255.0
    img_batch = tf.expand_dims(img_resized, 0)
    
    # Predict
    print("Running inference...")
    mask_pred = system.segmentation_model.predict(img_batch, verbose=0)[0]
    
    # Resize mask back
    mask_pred = tf.image.resize(mask_pred, img_original.shape[:2])
    mask_pred = mask_pred.numpy()
    
    # Convert to class labels
    mask_labels = np.argmax(mask_pred, axis=-1)
    
    # Create visualization
    vis = img_original.copy()
    
    # Overlay RBCs (green)
    rbc_mask = (mask_labels == 1)
    vis[rbc_mask] = vis[rbc_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    # Overlay WBCs (blue)
    wbc_mask = (mask_labels == 2)
    vis[wbc_mask] = vis[wbc_mask] * 0.5 + np.array([0, 0, 255]) * 0.5
    
    # Count cells
    rbc_count = np.sum(rbc_mask)
    wbc_count = np.sum(wbc_mask)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    test_name = Path(test_image).stem
    cv2.imwrite(str(output_path / f"{test_name}_original.png"), 
                cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_path / f"{test_name}_segmented.png"), 
                cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    # Save mask as colored image
    mask_colored = np.zeros_like(img_original)
    mask_colored[mask_labels == 1] = [0, 255, 0]  # RBC green
    mask_colored[mask_labels == 2] = [0, 0, 255]  # WBC blue
    cv2.imwrite(str(output_path / f"{test_name}_mask.png"), 
                cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR))
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nRBC pixels: {rbc_count:,}")
    print(f"WBC pixels: {wbc_count:,}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - {test_name}_original.png")
    print(f"  - {test_name}_segmented.png")
    print(f"  - {test_name}_mask.png")


def main():
    parser = argparse.ArgumentParser(
        description='Train segmentation model for blood cell detection'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train segmentation model')
    train_parser.add_argument('--data-dir', required=True,
                             help='Directory with images/ and masks/')
    train_parser.add_argument('--preset', default='balanced',
                             choices=['fast', 'balanced', 'accurate', 'research'],
                             help='Training preset')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=8,
                             help='Batch size')
    train_parser.add_argument('--resume-from', 
                             help='Path to checkpoint to resume from')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test segmentation model')
    test_parser.add_argument('--model', required=True,
                            help='Path to trained model')
    test_parser.add_argument('--image', required=True,
                            help='Test image path')
    test_parser.add_argument('--output-dir', default='./test_results',
                            help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_segmentation(
            data_dir=args.data_dir,
            config_preset=args.preset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume_from=args.resume_from
        )
    elif args.command == 'test':
        test_segmentation(
            model_path=args.model,
            test_image=args.image,
            output_dir=args.output_dir
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
