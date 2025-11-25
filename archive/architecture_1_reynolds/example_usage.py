"""
Example Usage Script
====================
Demonstrates how to use the Blood Cell Analysis System
"""

from blood_cell_system import BloodCellAnalysisSystem
from training_pipeline import TrainingPipeline
from inference_pipeline import InferencePipeline, batch_process_images
import os

def example_1_build_and_inspect():
    """Example 1: Build models and inspect architecture"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Building and Inspecting Models")
    print("="*80)
    
    # Create system
    system = BloodCellAnalysisSystem(
        seg_input_shape=(512, 512, 3),
        clf_input_shape=(130, 130, 3),
        d_reduced=64,
        use_domain_adapt=True
    )
    
    # Build models
    system.build_all_models()
    system.compile_models()
    
    # Print summaries
    print("\nSegmentation Model Parameters:", f"{system.segmentation_model.count_params():,}")
    print("RBC Classifier Parameters:", f"{system.rbc_classifier.count_params():,}")
    print("WBC Classifier Parameters:", f"{system.wbc_classifier.count_params():,}")
    
    print("\n✓ Models built successfully!")


def example_2_train_rbc_classifier():
    """Example 2: Train RBC classifier on pre-segmented data"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Training RBC Classifier")
    print("="*80)
    
    # This example assumes you have data in the following structure:
    # ./data/rbc_data/
    #   ├── healthy_RBC/
    #   ├── malaria_RBC/
    #   └── sickle_RBC/
    
    rbc_data_dir = './data/rbc_data'
    
    if not os.path.exists(rbc_data_dir):
        print(f"\n⚠️  Data directory not found: {rbc_data_dir}")
        print("Please create this directory structure:")
        print("  rbc_data/")
        print("    ├── healthy_RBC/")
        print("    ├── malaria_RBC/")
        print("    └── sickle_RBC/")
        return
    
    # Build system
    system = BloodCellAnalysisSystem(
        d_reduced=64,
        use_domain_adapt=True
    )
    system.build_all_models()
    system.compile_models()
    
    # Create training pipeline
    config = {'dataset_type': 'pre_segmented', 'use_domain_adaptation': True}
    pipeline = TrainingPipeline(system, config)
    
    # Train RBC classifier
    print("\nTraining RBC classifier...")
    pipeline.train_rbc_classifier(
        data_dir=rbc_data_dir,
        epochs=10,  # Use 50+ for production
        batch_size=32
    )
    
    # Save model
    system.save_models('./models/example_2')
    print("\n✓ Training complete! Model saved to ./models/example_2/")


def example_3_train_all_models():
    """Example 3: Train all models sequentially"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Training All Models")
    print("="*80)
    
    # Check if data directories exist
    seg_data = './data/segmentation'
    rbc_data = './data/rbc_data'
    wbc_data = './data/wbc_data'
    
    missing_dirs = []
    if not os.path.exists(rbc_data):
        missing_dirs.append(rbc_data)
    if not os.path.exists(wbc_data):
        missing_dirs.append(wbc_data)
    
    if missing_dirs:
        print(f"\n⚠️  Missing data directories: {', '.join(missing_dirs)}")
        print("\nPlease prepare your datasets first.")
        return
    
    # Build system
    system = BloodCellAnalysisSystem(
        d_reduced=64,
        use_domain_adapt=True
    )
    system.build_all_models()
    system.compile_models()
    
    # Create training pipeline
    config = {'dataset_type': 'pre_segmented', 'use_domain_adaptation': True}
    pipeline = TrainingPipeline(system, config)
    
    # Train all models
    pipeline.train_all_sequential(
        seg_data_dir=seg_data if os.path.exists(seg_data) else None,
        rbc_data_dir=rbc_data,
        wbc_data_dir=wbc_data,
        epochs_seg=10,   # Use 50+ for production
        epochs_rbc=10,   # Use 50+ for production
        epochs_wbc=10    # Use 50+ for production
    )
    
    print("\n✓ All models trained successfully!")


def example_4_inference_single_image():
    """Example 4: Run inference on a single image"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Single Image Inference")
    print("="*80)
    
    # Check if models exist
    model_dir = './final_models'
    if not os.path.exists(model_dir):
        print(f"\n⚠️  Models not found in {model_dir}")
        print("Please train models first (see examples 2 or 3)")
        return
    
    # Check if test image exists
    test_image = './test_images/sample.png'
    if not os.path.exists(test_image):
        print(f"\n⚠️  Test image not found: {test_image}")
        print("Please provide a test microscopy image")
        return
    
    # Load models
    print("Loading trained models...")
    system = BloodCellAnalysisSystem()
    system.load_models(model_dir)
    
    # Create inference pipeline
    pipeline = InferencePipeline(
        system=system,
        confidence_threshold=0.7,
        uncertain_margin=0.15
    )
    
    # Process image
    print(f"\nProcessing image: {test_image}")
    report = pipeline.process_image(
        test_image,
        save_visualization=True,
        output_dir='./results'
    )
    
    # Print summary
    print(report.to_summary_text())
    
    # Access detailed results
    print("\nDetailed Results:")
    print(f"  Total RBCs: {report.rbc_count}")
    print(f"  Malaria-infected: {report.malaria_rbc}")
    print(f"  Sickle cell: {report.sickle_rbc}")
    print(f"  Total WBCs: {report.wbc_count}")
    print(f"  Leukemia: {report.leukemia_wbc}")
    
    print("\n✓ Results saved to ./results/")


def example_5_batch_processing():
    """Example 5: Process multiple images in batch"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Batch Processing")
    print("="*80)
    
    # Check if models exist
    model_dir = './final_models'
    if not os.path.exists(model_dir):
        print(f"\n⚠️  Models not found in {model_dir}")
        print("Please train models first")
        return
    
    # Check if test images directory exists
    test_dir = './test_images'
    if not os.path.exists(test_dir):
        print(f"\n⚠️  Test images directory not found: {test_dir}")
        print("Please provide test images")
        return
    
    # Load models
    print("Loading trained models...")
    system = BloodCellAnalysisSystem()
    system.load_models(model_dir)
    
    # Create inference pipeline
    pipeline = InferencePipeline(system)
    
    # Batch process
    print(f"\nProcessing all images in: {test_dir}")
    batch_process_images(
        image_dir=test_dir,
        pipeline=pipeline,
        output_dir='./batch_results'
    )
    
    print("\n✓ Batch processing complete! Results saved to ./batch_results/")


def example_6_custom_configuration():
    """Example 6: Custom configuration for different use cases"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Custom Configurations")
    print("="*80)
    
    print("\n1. Fast Training Configuration (for testing)")
    print("-" * 40)
    fast_system = BloodCellAnalysisSystem(
        d_reduced=32,           # Lower dimension = faster
        use_domain_adapt=False  # Disable domain adaptation
    )
    print("   d_reduced: 32")
    print("   domain_adapt: False")
    print("   → Use for: Quick experiments")
    
    print("\n2. High Accuracy Configuration (for production)")
    print("-" * 40)
    accurate_system = BloodCellAnalysisSystem(
        d_reduced=128,          # Higher dimension = better accuracy
        use_domain_adapt=True   # Handle multiple staining methods
    )
    print("   d_reduced: 128")
    print("   domain_adapt: True")
    print("   → Use for: Final deployment")
    
    print("\n3. Balanced Configuration (recommended)")
    print("-" * 40)
    balanced_system = BloodCellAnalysisSystem(
        d_reduced=64,
        use_domain_adapt=True
    )
    print("   d_reduced: 64")
    print("   domain_adapt: True")
    print("   → Use for: Most cases")
    
    print("\n✓ Configuration examples complete!")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("BLOOD CELL ANALYSIS SYSTEM - EXAMPLES")
    print("="*80)
    
    print("\nAvailable examples:")
    print("  1. Build and inspect models")
    print("  2. Train RBC classifier")
    print("  3. Train all models")
    print("  4. Inference on single image")
    print("  5. Batch processing")
    print("  6. Custom configurations")
    
    print("\nRunning Example 1 (Build and Inspect)...")
    example_1_build_and_inspect()
    
    print("\n" + "="*80)
    print("To run other examples, uncomment them in the main() function")
    print("="*80)
    
    # Uncomment to run other examples:
    # example_2_train_rbc_classifier()
    # example_3_train_all_models()
    # example_4_inference_single_image()
    # example_5_batch_processing()
    # example_6_custom_configuration()


if __name__ == "__main__":
    main()
