"""
Complete Blood Cell Analysis System
====================================
Three-model architecture:
1. Segmentation Model: Detects and classifies cells as RBC/WBC
2. RBC Classifier: Classifies RBCs into healthy/malaria/sickle cell
3. WBC Classifier: Classifies WBCs into healthy/leukemia

Preserves Reynolds Networks concepts with domain adaptation for different staining methods.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple, List, Dict, Optional

# ============================================================================
# REYNOLDS NETWORKS CORE COMPONENTS (Preserved from original)
# ============================================================================

@tf.custom_gradient
def gradient_reversal(x):
    """Gradient reversal for domain adaptation"""
    def grad(dy):
        return -dy
    return x, grad


class GradientReversalLayer(layers.Layer):
    """Domain adaptation layer for handling different staining methods"""
    def __init__(self, lambda_val=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_val = lambda_val
    
    def call(self, x):
        return gradient_reversal(x * self.lambda_val)
    
    def get_config(self):
        config = super().get_config()
        config.update({"lambda_val": self.lambda_val})
        return config


class EfficientReynoldsFeatureOperator(layers.Layer):
    """
    Efficient Reynolds operator using cyclic transpositions (Theorem 3 from paper)
    Complexity: O(n) instead of O(n!)
    """
    def __init__(self, d_input, d_reduced, operator_type='cyclic_transpositions', **kwargs):
        super().__init__(**kwargs)
        self.d_input = d_input
        self.d_reduced = d_reduced
        self.operator_type = operator_type
        
    def build(self, input_shape):
        # Dimension reduction projection (512 -> d_reduced)
        self.projection = layers.Dense(self.d_reduced, use_bias=False, name='reynolds_projection')
        
        # Learnable weights for cyclic permutations
        self.perm_weights = self.add_weight(
            name='perm_weights',
            shape=(self.d_reduced,),
            initializer='ones',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, x):
        """
        Args:
            x: (batch, n, d_input) - feature maps
        Returns:
            (batch, n, d_reduced) - Reynolds averaged features
        """
        # Project to lower dimension
        x_proj = self.projection(x)  # (batch, n, d_reduced)
        
        # Apply cyclic transpositions (O(n) complexity)
        batch_size = tf.shape(x_proj)[0]
        n = tf.shape(x_proj)[1]
        
        # Stack all cyclic permutations
        rolled = []
        for i in range(tf.get_static_value(n) or 8):  # Default to 8 if dynamic
            rolled.append(tf.roll(x_proj, shift=-i, axis=1))
        
        # Average over all permutations
        x_reynolds = tf.reduce_mean(tf.stack(rolled, axis=0), axis=0)
        
        # Apply learnable weights
        x_reynolds = x_reynolds * self.perm_weights
        
        return x_reynolds
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_input": self.d_input,
            "d_reduced": self.d_reduced,
            "operator_type": self.operator_type
        })
        return config


class FeatureAttention(layers.Layer):
    """Multi-head attention for feature refinement"""
    def __init__(self, d_model, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        
    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            name='feature_attention'
        )
        self.norm = layers.LayerNormalization()
        super().build(input_shape)
    
    def call(self, x):
        """
        Args:
            x: (batch, n, d_model)
        Returns:
            (batch, n, d_model)
        """
        attn_out = self.attention(x, x)
        return self.norm(x + attn_out)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads
        })
        return config


class AdaptiveSetAggregation(layers.Layer):
    """Combines multiple aggregation strategies for set features"""
    def __init__(self, d_output, **kwargs):
        super().__init__(**kwargs)
        self.d_output = d_output
    
    def build(self, input_shape):
        d_input = input_shape[-1]
        
        # Learnable aggregation weights
        self.weight_net = keras.Sequential([
            layers.Dense(d_input // 2, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ], name='aggregation_weights')
        
        # Output projection
        self.output_proj = layers.Dense(self.d_output, name='aggregation_output')
        super().build(input_shape)
    
    def call(self, x):
        """
        Args:
            x: (batch, n, d_input)
        Returns:
            (batch, d_output)
        """
        # Weighted sum
        weights = self.weight_net(x)  # (batch, n, 1)
        weighted_sum = tf.reduce_sum(x * weights, axis=1)  # (batch, d_input)
        
        # Max pooling
        max_pool = tf.reduce_max(x, axis=1)  # (batch, d_input)
        
        # Mean pooling
        mean_pool = tf.reduce_mean(x, axis=1)  # (batch, d_input)
        
        # Concatenate all
        combined = tf.concat([weighted_sum, max_pool, mean_pool], axis=-1)
        
        # Project to output dimension
        return self.output_proj(combined)
    
    def get_config(self):
        config = super().get_config()
        config.update({"d_output": self.d_output})
        return config


def enhanced_conv_block(x, filters, strides=1, name=None):
    """Enhanced convolutional block with residual connection"""
    shortcut = x
    
    # Main path
    y = layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False, name=f'{name}_conv1' if name else None)(x)
    y = layers.BatchNormalization(name=f'{name}_bn1' if name else None)(y)
    y = layers.ReLU(name=f'{name}_relu1' if name else None)(y)
    
    y = layers.Conv2D(filters, 3, padding='same', use_bias=False, name=f'{name}_conv2' if name else None)(y)
    y = layers.BatchNormalization(name=f'{name}_bn2' if name else None)(y)
    
    # Shortcut connection
    if strides != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False, name=f'{name}_shortcut' if name else None)(x)
        shortcut = layers.BatchNormalization(name=f'{name}_bn_shortcut' if name else None)(shortcut)
    
    y = layers.Add(name=f'{name}_add' if name else None)([y, shortcut])
    y = layers.ReLU(name=f'{name}_relu2' if name else None)(y)
    
    return y


# ============================================================================
# COLOR NORMALIZATION FOR DIFFERENT STAINING METHODS
# ============================================================================

class StainNormalizationLayer(layers.Layer):
    """
    Normalizes different staining methods to focus on structural features
    Uses color space transformation and normalization
    """
    def __init__(self, method='adaptive', **kwargs):
        super().__init__(**kwargs)
        self.method = method
    
    def build(self, input_shape):
        # Learnable stain matrix (for different staining protocols)
        self.stain_matrix = self.add_weight(
            name='stain_matrix',
            shape=(3, 3),
            initializer=tf.keras.initializers.Identity(),
            trainable=True
        )
        
        # Learnable normalization parameters
        self.gamma = self.add_weight(
            name='gamma',
            shape=(3,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(3,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, x):
        """
        Args:
            x: (batch, h, w, 3) - RGB image
        Returns:
            (batch, h, w, 3) - normalized image
        """
        # Convert to optical density space
        x_od = -tf.math.log(x + 1e-6)
        
        # Apply stain matrix
        original_shape = tf.shape(x_od)
        x_flat = tf.reshape(x_od, [-1, 3])
        x_transformed = tf.matmul(x_flat, self.stain_matrix)
        x_transformed = tf.reshape(x_transformed, original_shape)
        
        # Normalize
        x_norm = x_transformed * self.gamma + self.beta
        
        # Convert back
        x_out = tf.exp(-x_norm)
        
        return tf.clip_by_value(x_out, 0, 1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"method": self.method})
        return config


# ============================================================================
# MODEL 1: SEGMENTATION MODEL (Detects RBC and WBC)
# ============================================================================

def build_segmentation_model(
    input_shape=(512, 512, 3),
    num_classes=3,  # Background, RBC, WBC
    use_stain_norm=True
):
    """
    U-Net based segmentation model with Reynolds features
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes (background, RBC, WBC)
        use_stain_norm: Whether to use stain normalization
    
    Returns:
        Keras model for segmentation
    """
    inputs = layers.Input(shape=input_shape, name='seg_input')
    
    # Stain normalization
    x = StainNormalizationLayer(name='stain_norm')(inputs) if use_stain_norm else inputs
    
    # Encoder
    # Block 1
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    # Block 2
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    # Block 3
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    
    # Block 4
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(2)(c4)
    
    # Bottleneck with Reynolds operator
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)
    
    # Apply Reynolds operator to bottleneck features
    # Reshape to (batch, n, d) for Reynolds operator
    h, w = tf.shape(c5)[1], tf.shape(c5)[2]
    c5_flat = layers.Reshape((-1, 1024))(c5)
    c5_reynolds = EfficientReynoldsFeatureOperator(1024, 256, name='seg_reynolds')(c5_flat)
    c5_reynolds = FeatureAttention(256, name='seg_attention')(c5_reynolds)
    c5_out = layers.Dense(1024, activation='relu', name='seg_dense')(c5_reynolds)
    c5_reshaped = layers.Reshape((h, w, 1024))(c5_out)
    
    # Decoder
    # Block 6
    u6 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(c5_reshaped)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(c6)
    
    # Block 7
    u7 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(c7)
    
    # Block 8
    u8 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(c8)
    
    # Block 9
    u9 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(c9)
    
    # Output
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', name='seg_output')(c9)
    
    model = models.Model(inputs, outputs, name='SegmentationModel')
    
    return model


# ============================================================================
# MODEL 2: RBC CLASSIFIER (Healthy, Malaria, Sickle Cell)
# ============================================================================

def build_rbc_classifier(
    input_shape=(130, 130, 3),
    num_classes=3,  # healthy_RBC, malaria_RBC, sickle_RBC
    d_reduced=64,
    use_domain_adapt=True
):
    """
    RBC classifier with Reynolds Networks and domain adaptation
    
    Args:
        input_shape: Input image shape for single cell
        num_classes: 3 (healthy, malaria, sickle cell)
        d_reduced: Reynolds dimension reduction
        use_domain_adapt: Use domain adaptation for different staining
    
    Returns:
        Keras model for RBC classification
    """
    inputs = layers.Input(shape=input_shape, name='rbc_input')
    
    # Stain normalization
    x = StainNormalizationLayer(name='rbc_stain_norm')(inputs)
    
    # CNN backbone
    y = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False, name='rbc_conv1')(x)
    y = layers.BatchNormalization(name='rbc_bn1')(y)
    y = layers.ReLU(name='rbc_relu1')(y)
    y = layers.MaxPooling2D(3, 2, padding='same', name='rbc_pool1')(y)
    
    # Residual blocks
    for i, f in enumerate([64, 64]):
        y = enhanced_conv_block(y, f, name=f'rbc_block1_{i}')
    
    for i, f in enumerate([128, 128]):
        y = enhanced_conv_block(y, f, strides=2 if i == 0 else 1, name=f'rbc_block2_{i}')
    
    for i, f in enumerate([256, 256]):
        y = enhanced_conv_block(y, f, strides=2 if i == 0 else 1, name=f'rbc_block3_{i}')
    
    # Feature maps
    fmap = layers.Conv2D(512, 3, padding='same', use_bias=False, name='rbc_fmap')(y)
    fmap = layers.BatchNormalization(name='rbc_fmap_bn')(fmap)
    fmap = layers.ReLU(name='rbc_fmap_relu')(fmap)
    
    # Reshape for Reynolds operator
    fs = layers.Reshape((-1, 512), name='rbc_reshape')(fmap)
    
    # Reynolds + Attention + Aggregation
    er = EfficientReynoldsFeatureOperator(512, d_reduced, name='rbc_reynolds')(fs)
    att = FeatureAttention(d_reduced, name='rbc_attention')(er)
    gf = AdaptiveSetAggregation(512, name='rbc_aggregation')(att)
    
    # Domain adaptation branch
    if use_domain_adapt:
        gf_reversed = GradientReversalLayer(lambda_val=0.1, name='rbc_grl')(gf)
        domain_output = layers.Dense(128, activation='relu', name='rbc_domain_hidden')(gf_reversed)
        domain_output = layers.Dense(1, activation='sigmoid', name='rbc_domain_output')(domain_output)
    
    # Classification head
    z = layers.Dropout(0.5, name='rbc_dropout1')(gf)
    z = layers.Dense(256, activation='relu', name='rbc_fc1')(z)
    z = layers.Dropout(0.3, name='rbc_dropout2')(z)
    class_output = layers.Dense(num_classes, activation='softmax', name='rbc_class_output')(z)
    
    # Build model
    if use_domain_adapt:
        model = models.Model(inputs, [class_output, domain_output], name='RBCClassifier')
    else:
        model = models.Model(inputs, class_output, name='RBCClassifier')
    
    return model


# ============================================================================
# MODEL 3: WBC CLASSIFIER (Healthy, Leukemia)
# ============================================================================

def build_wbc_classifier(
    input_shape=(130, 130, 3),
    num_classes=2,  # healthy_WBC, cancer_WBC
    d_reduced=64,
    use_domain_adapt=True
):
    """
    WBC classifier with Reynolds Networks and domain adaptation
    
    Args:
        input_shape: Input image shape for single cell
        num_classes: 2 (healthy, leukemia)
        d_reduced: Reynolds dimension reduction
        use_domain_adapt: Use domain adaptation for different staining
    
    Returns:
        Keras model for WBC classification
    """
    inputs = layers.Input(shape=input_shape, name='wbc_input')
    
    # Stain normalization
    x = StainNormalizationLayer(name='wbc_stain_norm')(inputs)
    
    # CNN backbone
    y = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False, name='wbc_conv1')(x)
    y = layers.BatchNormalization(name='wbc_bn1')(y)
    y = layers.ReLU(name='wbc_relu1')(y)
    y = layers.MaxPooling2D(3, 2, padding='same', name='wbc_pool1')(y)
    
    # Residual blocks
    for i, f in enumerate([64, 64]):
        y = enhanced_conv_block(y, f, name=f'wbc_block1_{i}')
    
    for i, f in enumerate([128, 128]):
        y = enhanced_conv_block(y, f, strides=2 if i == 0 else 1, name=f'wbc_block2_{i}')
    
    for i, f in enumerate([256, 256]):
        y = enhanced_conv_block(y, f, strides=2 if i == 0 else 1, name=f'wbc_block3_{i}')
    
    # Feature maps
    fmap = layers.Conv2D(512, 3, padding='same', use_bias=False, name='wbc_fmap')(y)
    fmap = layers.BatchNormalization(name='wbc_fmap_bn')(fmap)
    fmap = layers.ReLU(name='wbc_fmap_relu')(fmap)
    
    # Reshape for Reynolds operator
    fs = layers.Reshape((-1, 512), name='wbc_reshape')(fmap)
    
    # Reynolds + Attention + Aggregation
    er = EfficientReynoldsFeatureOperator(512, d_reduced, name='wbc_reynolds')(fs)
    att = FeatureAttention(d_reduced, name='wbc_attention')(er)
    gf = AdaptiveSetAggregation(512, name='wbc_aggregation')(att)
    
    # Domain adaptation branch
    if use_domain_adapt:
        gf_reversed = GradientReversalLayer(lambda_val=0.1, name='wbc_grl')(gf)
        domain_output = layers.Dense(128, activation='relu', name='wbc_domain_hidden')(gf_reversed)
        domain_output = layers.Dense(1, activation='sigmoid', name='wbc_domain_output')(domain_output)
    
    # Classification head
    z = layers.Dropout(0.5, name='wbc_dropout1')(gf)
    z = layers.Dense(256, activation='relu', name='wbc_fc1')(z)
    z = layers.Dropout(0.3, name='wbc_dropout2')(z)
    class_output = layers.Dense(num_classes, activation='softmax', name='wbc_class_output')(z)
    
    # Build model
    if use_domain_adapt:
        model = models.Model(inputs, [class_output, domain_output], name='WBCClassifier')
    else:
        model = models.Model(inputs, class_output, name='WBCClassifier')
    
    return model


# ============================================================================
# COMPLETE SYSTEM BUILDER
# ============================================================================

class BloodCellAnalysisSystem:
    """
    Complete end-to-end blood cell analysis system
    Combines segmentation and classification models
    """
    def __init__(
        self,
        seg_input_shape=(512, 512, 3),
        clf_input_shape=(130, 130, 3),
        d_reduced=64,
        use_domain_adapt=True
    ):
        self.seg_input_shape = seg_input_shape
        self.clf_input_shape = clf_input_shape
        self.d_reduced = d_reduced
        self.use_domain_adapt = use_domain_adapt
        
        # Build models
        self.segmentation_model = None
        self.rbc_classifier = None
        self.wbc_classifier = None
        
    def build_all_models(self):
        """Build all three models"""
        print("Building Segmentation Model...")
        self.segmentation_model = build_segmentation_model(
            input_shape=self.seg_input_shape,
            num_classes=3  # Background, RBC, WBC
        )
        
        print("Building RBC Classifier...")
        self.rbc_classifier = build_rbc_classifier(
            input_shape=self.clf_input_shape,
            num_classes=3,  # healthy, malaria, sickle_cell
            d_reduced=self.d_reduced,
            use_domain_adapt=self.use_domain_adapt
        )
        
        print("Building WBC Classifier...")
        self.wbc_classifier = build_wbc_classifier(
            input_shape=self.clf_input_shape,
            num_classes=2,  # healthy, leukemia
            d_reduced=self.d_reduced,
            use_domain_adapt=self.use_domain_adapt
        )
        
        print("All models built successfully!")
        return self
    
    def compile_models(self, learning_rate=1e-4):
        """Compile all models with appropriate losses"""
        # Segmentation model
        self.segmentation_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.MeanIoU(num_classes=3)]
        )
        
        # RBC classifier
        if self.use_domain_adapt:
            self.rbc_classifier.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss={
                    'rbc_class_output': 'categorical_crossentropy',
                    'rbc_domain_output': 'binary_crossentropy'
                },
                loss_weights={
                    'rbc_class_output': 1.0,
                    'rbc_domain_output': 0.1
                },
                metrics={
                    'rbc_class_output': 'accuracy',
                    'rbc_domain_output': 'accuracy'
                }
            )
        else:
            self.rbc_classifier.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # WBC classifier
        if self.use_domain_adapt:
            self.wbc_classifier.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss={
                    'wbc_class_output': 'categorical_crossentropy',
                    'wbc_domain_output': 'binary_crossentropy'
                },
                loss_weights={
                    'wbc_class_output': 1.0,
                    'wbc_domain_output': 0.1
                },
                metrics={
                    'wbc_class_output': 'accuracy',
                    'wbc_domain_output': 'accuracy'
                }
            )
        else:
            self.wbc_classifier.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        print("All models compiled successfully!")
        return self
    
    def summary(self):
        """Print summaries of all models"""
        print("\n" + "="*80)
        print("SEGMENTATION MODEL")
        print("="*80)
        self.segmentation_model.summary()
        
        print("\n" + "="*80)
        print("RBC CLASSIFIER")
        print("="*80)
        self.rbc_classifier.summary()
        
        print("\n" + "="*80)
        print("WBC CLASSIFIER")
        print("="*80)
        self.wbc_classifier.summary()
    
    def save_models(self, save_dir='./models'):
        """Save all models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        self.segmentation_model.save(f'{save_dir}/segmentation_model.keras')
        self.rbc_classifier.save(f'{save_dir}/rbc_classifier.keras')
        self.wbc_classifier.save(f'{save_dir}/wbc_classifier.keras')
        
        print(f"Models saved to {save_dir}/")
    
    def load_models(self, save_dir='./models'):
        """Load all models"""
        self.segmentation_model = keras.models.load_model(
            f'{save_dir}/segmentation_model.keras',
            custom_objects={
                'StainNormalizationLayer': StainNormalizationLayer,
                'EfficientReynoldsFeatureOperator': EfficientReynoldsFeatureOperator,
                'FeatureAttention': FeatureAttention,
                'AdaptiveSetAggregation': AdaptiveSetAggregation
            }
        )
        
        self.rbc_classifier = keras.models.load_model(
            f'{save_dir}/rbc_classifier.keras',
            custom_objects={
                'StainNormalizationLayer': StainNormalizationLayer,
                'GradientReversalLayer': GradientReversalLayer,
                'EfficientReynoldsFeatureOperator': EfficientReynoldsFeatureOperator,
                'FeatureAttention': FeatureAttention,
                'AdaptiveSetAggregation': AdaptiveSetAggregation
            }
        )
        
        self.wbc_classifier = keras.models.load_model(
            f'{save_dir}/wbc_classifier.keras',
            custom_objects={
                'StainNormalizationLayer': StainNormalizationLayer,
                'GradientReversalLayer': GradientReversalLayer,
                'EfficientReynoldsFeatureOperator': EfficientReynoldsFeatureOperator,
                'FeatureAttention': FeatureAttention,
                'AdaptiveSetAggregation': AdaptiveSetAggregation
            }
        )
        
        print(f"Models loaded from {save_dir}/")


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("Building Blood Cell Analysis System...")
    
    system = BloodCellAnalysisSystem(
        seg_input_shape=(512, 512, 3),
        clf_input_shape=(130, 130, 3),
        d_reduced=64,
        use_domain_adapt=True
    )
    
    system.build_all_models()
    system.compile_models()
    
    print("\n" + "="*80)
    print("SYSTEM READY!")
    print("="*80)
    print("\nModels:")
    print(f"  1. Segmentation Model: {system.segmentation_model.count_params():,} parameters")
    print(f"  2. RBC Classifier: {system.rbc_classifier.count_params():,} parameters")
    print(f"  3. WBC Classifier: {system.wbc_classifier.count_params():,} parameters")
    print("\nFeatures:")
    print("  ✓ Reynolds Networks (O(n) complexity)")
    print("  ✓ Domain Adaptation (handles different staining)")
    print("  ✓ Feature Attention")
    print("  ✓ Stain Normalization")
    print("  ✓ End-to-end pipeline ready")
