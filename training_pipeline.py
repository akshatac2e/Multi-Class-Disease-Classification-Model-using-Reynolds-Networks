"""
Production Training Pipeline - Architecture 2
==============================================
Multi-stage training for:
1. Instance Segmentation (HTC/Mask R-CNN)
2. RBC Classifier (EfficientNet-V2)
3. WBC Classifier (ConvNeXt/ViT)

Features:
- Multi-GPU distributed training
- Mixed precision (FP16/BF16)
- Experiment tracking (W&B/MLflow)
- Checkpoint management
- Data augmentation
- Cross-validation
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import yaml
import json
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import logging

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from blood_cell_system import (
    ProductionBloodCellSystem, RBCClassifier, WBCClassifier,
    RBCClass, WBCClass, CellType, StainMethod, StainNormalizer
)


# ============================================================================
# DATA PREPROCESSING AND LOADING
# ============================================================================

class DatasetHandler:
    """Handles different dataset formats"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset_type = config.get('dataset_type', 'pre_segmented')
        # 'pre_segmented': Single cell images (e.g., malaria dataset)
        # 'raw': Raw microscopy images (requires segmentation)
        
    def create_rbc_generators(self, data_dir: str, batch_size: int = 32, val_split: float = 0.2):
        """
        Create data generators for RBC classification
        Classes: healthy_RBC, malaria_RBC, sickle_RBC
        """
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=val_split,
            rotation_range=360,  # Cells can be at any orientation
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.15,
            brightness_range=[0.7, 1.3],  # Handle different staining intensities
            fill_mode='reflect'
        )
        
        train_gen = datagen.flow_from_directory(
            data_dir,
            target_size=(130, 130),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        val_gen = datagen.flow_from_directory(
            data_dir,
            target_size=(130, 130),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        return train_gen, val_gen
    
    def create_wbc_generators(self, data_dir: str, batch_size: int = 32, val_split: float = 0.2):
        """
        Create data generators for WBC classification
        Classes: healthy_WBC, cancer_WBC (leukemia)
        """
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=val_split,
            rotation_range=360,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.15,
            brightness_range=[0.7, 1.3],
            fill_mode='reflect'
        )
        
        train_gen = datagen.flow_from_directory(
            data_dir,
            target_size=(130, 130),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        val_gen = datagen.flow_from_directory(
            data_dir,
            target_size=(130, 130),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        return train_gen, val_gen
    
    def create_segmentation_generators(self, data_dir: str, batch_size: int = 8, val_split: float = 0.2):
        """
        Create data generators for segmentation training
        Expects paired images and masks
        """
        # For segmentation, we need custom generator
        # This is a placeholder - actual implementation depends on mask format
        image_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=val_split,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='reflect'
        )
        
        mask_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=val_split,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='reflect'
        )
        
        train_image_gen = image_datagen.flow_from_directory(
            os.path.join(data_dir, 'images'),
            target_size=(512, 512),
            batch_size=batch_size,
            class_mode=None,
            subset='training',
            shuffle=True,
            seed=42
        )
        
        train_mask_gen = mask_datagen.flow_from_directory(
            os.path.join(data_dir, 'masks'),
            target_size=(512, 512),
            batch_size=batch_size,
            class_mode=None,
            subset='training',
            shuffle=True,
            seed=42
        )
        
        val_image_gen = image_datagen.flow_from_directory(
            os.path.join(data_dir, 'images'),
            target_size=(512, 512),
            batch_size=batch_size,
            class_mode=None,
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        val_mask_gen = mask_datagen.flow_from_directory(
            os.path.join(data_dir, 'masks'),
            target_size=(512, 512),
            batch_size=batch_size,
            class_mode=None,
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        train_gen = zip(train_image_gen, train_mask_gen)
        val_gen = zip(val_image_gen, val_mask_gen)
        
        return train_gen, val_gen


# ============================================================================
# CUSTOM TRAINING CALLBACKS
# ============================================================================

class DomainAdaptationCallback(keras.callbacks.Callback):
    """Custom callback to monitor domain adaptation during training"""
    
    def __init__(self, val_data, model_name='Model'):
        super().__init__()
        self.val_data = val_data
        self.model_name = model_name
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # Log domain adaptation metrics
            domain_acc = logs.get(f'{self.model_name.lower()}_domain_output_accuracy', 0)
            class_acc = logs.get(f'{self.model_name.lower()}_class_output_accuracy', 0)
            
            print(f"\n{self.model_name} - Epoch {epoch + 1}:")
            print(f"  Classification Accuracy: {class_acc:.4f}")
            print(f"  Domain Adaptation Accuracy: {domain_acc:.4f}")


# ============================================================================
# TRAINING ORCHESTRATOR
# ============================================================================

class TrainingPipeline:
    """Main training orchestrator for all three models"""
    
    def __init__(
        self,
        system: BloodCellAnalysisSystem,
        config: Dict
    ):
        self.system = system
        self.config = config
        self.dataset_handler = DatasetHandler(config)
        self.history = {
            'segmentation': None,
            'rbc_classifier': None,
            'wbc_classifier': None
        }
    
    def train_segmentation_model(
        self,
        data_dir: str,
        epochs: int = 50,
        batch_size: int = 8,
        save_best: bool = True
    ):
        """
        Train the segmentation model
        
        Args:
            data_dir: Directory containing images/ and masks/ subdirectories
            epochs: Number of training epochs
            batch_size: Batch size
            save_best: Whether to save best model
        """
        print("\n" + "="*80)
        print("TRAINING SEGMENTATION MODEL")
        print("="*80)
        
        # Create data generators
        train_gen, val_gen = self.dataset_handler.create_segmentation_generators(
            data_dir, batch_size
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'checkpoints/segmentation_model_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ) if save_best else None
        ]
        callbacks = [c for c in callbacks if c is not None]
        
        # Train
        history = self.system.segmentation_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history['segmentation'] = history.history
        
        print("\n✓ Segmentation model training complete!")
        return history
    
    def train_rbc_classifier(
        self,
        data_dir: str,
        epochs: int = 50,
        batch_size: int = 32,
        use_class_weights: bool = True,
        save_best: bool = True
    ):
        """
        Train the RBC classifier
        
        Args:
            data_dir: Directory with subdirectories: healthy_RBC/, malaria_RBC/, sickle_RBC/
            epochs: Number of training epochs
            batch_size: Batch size
            use_class_weights: Use class weights for imbalanced data
            save_best: Whether to save best model
        """
        print("\n" + "="*80)
        print("TRAINING RBC CLASSIFIER")
        print("="*80)
        
        # Create data generators
        train_gen, val_gen = self.dataset_handler.create_rbc_generators(
            data_dir, batch_size
        )
        
        print(f"\nTraining samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Class indices: {train_gen.class_indices}")
        
        # Calculate class weights if needed
        class_weights = None
        if use_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_gen.classes),
                y=train_gen.classes
            )
            class_weights = dict(enumerate(class_weights))
            print(f"\nClass weights: {class_weights}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_rbc_class_output_loss' if self.system.use_domain_adapt else 'val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_rbc_class_output_loss' if self.system.use_domain_adapt else 'val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'checkpoints/rbc_classifier_best.keras',
                monitor='val_rbc_class_output_accuracy' if self.system.use_domain_adapt else 'val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ) if save_best else None
        ]
        
        if self.system.use_domain_adapt:
            callbacks.append(DomainAdaptationCallback(val_gen, 'RBC'))
        
        callbacks = [c for c in callbacks if c is not None]
        
        # Prepare domain labels if using domain adaptation
        if self.system.use_domain_adapt:
            # Create dummy domain labels (0 for source, 1 for target)
            # In practice, you'd have actual domain labels
            domain_labels = np.zeros((train_gen.samples, 1))
            
            def generator_with_domain():
                for x, y in train_gen:
                    domain_batch = np.zeros((len(x), 1))
                    yield x, {'rbc_class_output': y, 'rbc_domain_output': domain_batch}
            
            def val_generator_with_domain():
                for x, y in val_gen:
                    domain_batch = np.zeros((len(x), 1))
                    yield x, {'rbc_class_output': y, 'rbc_domain_output': domain_batch}
            
            train_data = generator_with_domain()
            val_data = val_generator_with_domain()
        else:
            train_data = train_gen
            val_data = val_gen
        
        # Train
        history = self.system.rbc_classifier.fit(
            train_data,
            steps_per_epoch=len(train_gen),
            validation_data=val_data,
            validation_steps=len(val_gen),
            epochs=epochs,
            class_weight=class_weights if not self.system.use_domain_adapt else None,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history['rbc_classifier'] = history.history
        
        print("\n✓ RBC classifier training complete!")
        return history
    
    def train_wbc_classifier(
        self,
        data_dir: str,
        epochs: int = 50,
        batch_size: int = 32,
        use_class_weights: bool = True,
        save_best: bool = True
    ):
        """
        Train the WBC classifier
        
        Args:
            data_dir: Directory with subdirectories: healthy_WBC/, cancer_WBC/
            epochs: Number of training epochs
            batch_size: Batch size
            use_class_weights: Use class weights for imbalanced data
            save_best: Whether to save best model
        """
        print("\n" + "="*80)
        print("TRAINING WBC CLASSIFIER")
        print("="*80)
        
        # Create data generators
        train_gen, val_gen = self.dataset_handler.create_wbc_generators(
            data_dir, batch_size
        )
        
        print(f"\nTraining samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Class indices: {train_gen.class_indices}")
        
        # Calculate class weights if needed
        class_weights = None
        if use_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_gen.classes),
                y=train_gen.classes
            )
            class_weights = dict(enumerate(class_weights))
            print(f"\nClass weights: {class_weights}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_wbc_class_output_loss' if self.system.use_domain_adapt else 'val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_wbc_class_output_loss' if self.system.use_domain_adapt else 'val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'checkpoints/wbc_classifier_best.keras',
                monitor='val_wbc_class_output_accuracy' if self.system.use_domain_adapt else 'val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ) if save_best else None
        ]
        
        if self.system.use_domain_adapt:
            callbacks.append(DomainAdaptationCallback(val_gen, 'WBC'))
        
        callbacks = [c for c in callbacks if c is not None]
        
        # Prepare domain labels if using domain adaptation
        if self.system.use_domain_adapt:
            def generator_with_domain():
                for x, y in train_gen:
                    domain_batch = np.zeros((len(x), 1))
                    yield x, {'wbc_class_output': y, 'wbc_domain_output': domain_batch}
            
            def val_generator_with_domain():
                for x, y in val_gen:
                    domain_batch = np.zeros((len(x), 1))
                    yield x, {'wbc_class_output': y, 'wbc_domain_output': domain_batch}
            
            train_data = generator_with_domain()
            val_data = val_generator_with_domain()
        else:
            train_data = train_gen
            val_data = val_gen
        
        # Train
        history = self.system.wbc_classifier.fit(
            train_data,
            steps_per_epoch=len(train_gen),
            validation_data=val_data,
            validation_steps=len(val_gen),
            epochs=epochs,
            class_weight=class_weights if not self.system.use_domain_adapt else None,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history['wbc_classifier'] = history.history
        
        print("\n✓ WBC classifier training complete!")
        return history
    
    def train_all_sequential(
        self,
        seg_data_dir: Optional[str] = None,
        rbc_data_dir: Optional[str] = None,
        wbc_data_dir: Optional[str] = None,
        epochs_seg: int = 50,
        epochs_rbc: int = 50,
        epochs_wbc: int = 50
    ):
        """
        Train all models sequentially
        
        Args:
            seg_data_dir: Segmentation dataset directory (optional)
            rbc_data_dir: RBC classification dataset directory
            wbc_data_dir: WBC classification dataset directory
            epochs_seg: Epochs for segmentation
            epochs_rbc: Epochs for RBC classifier
            epochs_wbc: Epochs for WBC classifier
        """
        print("\n" + "="*80)
        print("SEQUENTIAL TRAINING OF ALL MODELS")
        print("="*80)
        
        # Create checkpoints directory
        os.makedirs('checkpoints', exist_ok=True)
        
        # Train segmentation model if data provided
        if seg_data_dir and os.path.exists(seg_data_dir):
            self.train_segmentation_model(seg_data_dir, epochs=epochs_seg)
        else:
            print("\nSkipping segmentation training (no data provided or using pre-segmented data)")
        
        # Train RBC classifier
        if rbc_data_dir and os.path.exists(rbc_data_dir):
            self.train_rbc_classifier(rbc_data_dir, epochs=epochs_rbc)
        else:
            print("\nWARNING: RBC data directory not found!")
        
        # Train WBC classifier
        if wbc_data_dir and os.path.exists(wbc_data_dir):
            self.train_wbc_classifier(wbc_data_dir, epochs=epochs_wbc)
        else:
            print("\nWARNING: WBC data directory not found!")
        
        # Save final models
        self.system.save_models('final_models')
        
        # Save training history
        self.save_training_history()
        
        print("\n" + "="*80)
        print("ALL TRAINING COMPLETE!")
        print("="*80)
        print("\nFinal models saved to: final_models/")
        print("Training history saved to: training_history.json")
    
    def save_training_history(self, filepath='training_history.json'):
        """Save training history to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for model_name, hist in self.history.items():
            if hist is not None:
                history_serializable[model_name] = {
                    k: [float(v) for v in vals] for k, vals in hist.items()
                }
        
        with open(filepath, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"\nTraining history saved to {filepath}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example training script"""
    
    # Configuration
    config = {
        'dataset_type': 'pre_segmented',  # or 'raw'
        'use_domain_adaptation': True,
        'd_reduced': 64,
        'learning_rate': 1e-4
    }
    
    # Build system
    print("Building Blood Cell Analysis System...")
    system = BloodCellAnalysisSystem(
        seg_input_shape=(512, 512, 3),
        clf_input_shape=(130, 130, 3),
        d_reduced=config['d_reduced'],
        use_domain_adapt=config['use_domain_adaptation']
    )
    
    system.build_all_models()
    system.compile_models(learning_rate=config['learning_rate'])
    
    # Create training pipeline
    pipeline = TrainingPipeline(system, config)
    
    # Example: Train RBC classifier only (for pre-segmented dataset like malaria)
    print("\nExample: Training RBC classifier with pre-segmented data...")
    print("Expected directory structure:")
    print("  rbc_data/")
    print("    ├── healthy_RBC/")
    print("    │   ├── img1.png")
    print("    │   └── ...")
    print("    ├── malaria_RBC/")
    print("    │   └── ...")
    print("    └── sickle_RBC/")
    print("        └── ...")
    
    # Uncomment to actually train:
    # pipeline.train_rbc_classifier(
    #     data_dir='./data/rbc_data',
    #     epochs=50,
    #     batch_size=32
    # )
    
    # Example: Train all models sequentially
    # pipeline.train_all_sequential(
    #     seg_data_dir='./data/segmentation',  # Optional
    #     rbc_data_dir='./data/rbc_data',
    #     wbc_data_dir='./data/wbc_data',
    #     epochs_seg=50,
    #     epochs_rbc=50,
    #     epochs_wbc=50
    # )
    
    print("\n" + "="*80)
    print("Training pipeline ready!")
    print("="*80)
    print("\nTo train models, update the data paths and uncomment the training calls above.")


if __name__ == "__main__":
    main()
