# 🔬 Segmentation Model Training Guide

Complete guide to training the segmentation model for RBC and WBC detection.

## 📊 Recommended Datasets

### 🥇 Option 1: BCCD Dataset (Easiest)

**Best for**: Quick start, general blood cell detection

- **Source**: https://github.com/Shenggan/BCCD_Dataset
- **Size**: ~360 images
- **Format**: Pascal VOC XML annotations
- **Includes**: RBC, WBC, Platelets
- **Download**:
  ```bash
  git clone https://github.com/Shenggan/BCCD_Dataset.git
  ```

**Pros**:
- ✅ Free and publicly available
- ✅ Good variety of cell types
- ✅ Easy to process
- ✅ Community standard

**Cons**:
- ⚠️ Limited size (need augmentation)
- ⚠️ Bounding boxes only (need conversion to masks)

### 🥈 Option 2: Kaggle Blood Cell Detection

**Best for**: More variety, larger dataset

- **Source**: https://www.kaggle.com/datasets/draaslan/blood-cell-detection-dataset
- **Size**: ~364 images
- **Format**: XML annotations
- **Quality**: Multiple staining methods

**Download**:
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API token (get from kaggle.com/account)
# Place kaggle.json in ~/.kaggle/

# Download
kaggle datasets download -d draaslan/blood-cell-detection-dataset
unzip blood-cell-detection-dataset.zip
```

### 🥉 Option 3: Combined Dataset (Best Results)

**Recommended**: Combine multiple sources for robust model

1. BCCD Dataset (~360 images)
2. Kaggle Blood Cell Detection (~364 images)
3. Your own images (if available)

**Total**: 700+ images → With augmentation: 2,800+ images

### 📚 Additional Datasets

| Dataset | Size | Type | Source |
|---------|------|------|--------|
| **LISC** | 400+ | WBC segmentation | [Link](https://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.htm) |
| **ALL-IDB** | 260+ | Leukemia images | [Link](https://homes.di.unimi.it/scotti/all/) |
| **CellavisionDB** | Commercial | Professional | Requires license |

## 🛠️ Step-by-Step Training Process

### Step 1: Download and Prepare Data

#### Option A: Using BCCD Dataset

```bash
# 1. Clone dataset
git clone https://github.com/Shenggan/BCCD_Dataset.git

# 2. Convert to segmentation format
python prepare_segmentation_data.py \
    --mode bccd \
    --input ./BCCD_Dataset \
    --output ./data/segmentation

# 3. Verify dataset
python prepare_segmentation_data.py \
    --mode verify \
    --input ./data/segmentation
```

#### Option B: Augment Existing Dataset

```bash
# Increase dataset size 4x (original + 3 augmented versions)
python prepare_segmentation_data.py \
    --mode augment \
    --input ./data/segmentation \
    --output ./data/segmentation_augmented \
    --aug-factor 3
```

### Step 2: Verify Data Structure

Your data should look like:

```
data/segmentation/
├── images/
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── masks/
    ├── img_001.png  # Must match image filename
    ├── img_002.png
    └── ...
```

**Mask format**:
- RGB image (3 channels)
- Channel 0 (Red): Background = 255
- Channel 1 (Green): RBC = 255
- Channel 2 (Blue): WBC = 255

### Step 3: Train the Model

#### Quick Training (Testing)

```bash
python train_segmentation.py train \
    --data-dir ./data/segmentation \
    --preset fast \
    --epochs 10 \
    --batch-size 8
```

**Time**: ~30 minutes  
**Expected IoU**: 0.70-0.75

#### Recommended Training (Production)

```bash
python train_segmentation.py train \
    --data-dir ./data/segmentation_augmented \
    --preset balanced \
    --epochs 50 \
    --batch-size 8
```

**Time**: ~2-3 hours  
**Expected IoU**: 0.80-0.88

#### High Accuracy Training

```bash
python train_segmentation.py train \
    --data-dir ./data/segmentation_augmented \
    --preset accurate \
    --epochs 100 \
    --batch-size 4
```

**Time**: ~6-8 hours  
**Expected IoU**: 0.85-0.92

### Step 4: Test the Model

```bash
python train_segmentation.py test \
    --model checkpoints/segmentation_model_best.keras \
    --image test_images/sample.png \
    --output-dir ./test_results
```

This creates:
- `sample_original.png` - Original image
- `sample_segmented.png` - Segmented with overlays
- `sample_mask.png` - Pure segmentation mask

### Step 5: Integrate with Full Pipeline

```python
from blood_cell_system import BloodCellAnalysisSystem
from inference_pipeline import InferencePipeline

# Load trained segmentation model
system = BloodCellAnalysisSystem()
system.segmentation_model = tf.keras.models.load_model(
    'checkpoints/segmentation_model_best.keras'
)

# Load your RBC and WBC classifiers
system.rbc_classifier = tf.keras.models.load_model('models/rbc_classifier.keras')
system.wbc_classifier = tf.keras.models.load_model('models/wbc_classifier.keras')

# Run end-to-end inference
pipeline = InferencePipeline(system)
report = pipeline.process_image('microscopy_image.png')
print(report.to_summary_text())
```

## ⚙️ Configuration Options

### Presets

| Preset | d_reduced | Epochs | Batch Size | Time | IoU |
|--------|-----------|--------|------------|------|-----|
| **fast** | 32 | 10 | 16 | 30 min | 0.70-0.75 |
| **balanced** | 64 | 50 | 8 | 2-3 hrs | 0.80-0.88 |
| **accurate** | 128 | 100 | 4 | 6-8 hrs | 0.85-0.92 |
| **research** | 128 | 150 | 4 | 10-12 hrs | 0.88-0.95 |

### Training Parameters

```python
# Custom training
python train_segmentation.py train \
    --data-dir ./data/segmentation \
    --preset balanced \
    --epochs 50 \
    --batch-size 8
```

**Adjust batch size** if you get OOM errors:
- GPU 16GB: batch_size=8 (default)
- GPU 8GB: batch_size=4
- GPU 4GB: batch_size=2
- CPU only: batch_size=1 (very slow)

### Resume Training

```bash
# Resume from checkpoint
python train_segmentation.py train \
    --data-dir ./data/segmentation \
    --preset balanced \
    --epochs 100 \
    --resume-from checkpoints/segmentation_model_best.keras
```

## 📈 Expected Results

### Good Performance Indicators

✅ **Validation IoU > 0.85**: Excellent segmentation  
✅ **Validation IoU > 0.75**: Good, usable for production  
✅ **Training loss decreasing**: Model is learning  
✅ **Val loss stable**: No overfitting

### Poor Performance Indicators

❌ **Validation IoU < 0.70**: Need more data or longer training  
❌ **Val loss increasing**: Overfitting (reduce epochs or add augmentation)  
❌ **Loss not decreasing**: Learning rate too high/low or data issues

## 🎯 Tips for Best Results

### 1. Data Quality

- **Use high-quality images**: Clear cell boundaries
- **Consistent staining**: Or enable domain adaptation
- **Accurate masks**: Quality over quantity
- **Balanced classes**: Similar amounts of RBC/WBC examples

### 2. Data Augmentation

Recommended augmentations (already included):
- ✅ Rotation (360°)
- ✅ Flips (horizontal/vertical)
- ✅ Brightness/contrast (0.7-1.3×)
- ✅ Gaussian noise
- ✅ Elastic deformations

### 3. Training Strategy

**Start small, scale up**:
1. Train 10 epochs on small dataset (validate approach)
2. Train 50 epochs on augmented dataset (get baseline)
3. Train 100+ epochs for production (fine-tune)

**Monitor metrics**:
```bash
# Watch training progress
tensorboard --logdir logs/
```

### 4. Dataset Size Guidelines

| Images | With Augmentation | Expected IoU |
|--------|------------------|--------------|
| 100 | 400 | 0.70-0.75 |
| 200 | 800 | 0.75-0.82 |
| 500 | 2,000 | 0.82-0.88 |
| 1,000+ | 4,000+ | 0.88-0.95 |

## 🔧 Troubleshooting

### Issue 1: Low IoU (< 0.70)

**Solutions**:
```bash
# 1. More augmentation
python prepare_segmentation_data.py --mode augment --aug-factor 5

# 2. Longer training
python train_segmentation.py train --epochs 100

# 3. Check data quality
python prepare_segmentation_data.py --mode verify --input ./data/segmentation
```

### Issue 2: Out of Memory

**Solutions**:
```bash
# Reduce batch size
python train_segmentation.py train --batch-size 4  # or 2 or 1

# Use smaller model
python train_segmentation.py train --preset fast
```

### Issue 3: Overfitting

**Symptoms**: Train IoU high, Val IoU low

**Solutions**:
- Add more data augmentation
- Reduce epochs
- Add dropout (modify model)
- Get more training data

### Issue 4: Poor Cell Separation

**If RBCs and WBCs overlap**:
- Use better annotation tools (CVAT, LabelMe)
- Draw circular/elliptical masks (not rectangles)
- Ensure clear boundaries in masks
- Add more difficult examples to training set

## 📝 Complete Example Workflow

```bash
# 1. Download BCCD dataset
git clone https://github.com/Shenggan/BCCD_Dataset.git

# 2. Convert to segmentation format
python prepare_segmentation_data.py \
    --mode bccd \
    --input ./BCCD_Dataset \
    --output ./data/segmentation

# 3. Augment dataset (3× increase)
python prepare_segmentation_data.py \
    --mode augment \
    --input ./data/segmentation \
    --output ./data/segmentation_augmented \
    --aug-factor 3

# 4. Verify dataset
python prepare_segmentation_data.py \
    --mode verify \
    --input ./data/segmentation_augmented

# 5. Train model (balanced preset, ~2-3 hours)
python train_segmentation.py train \
    --data-dir ./data/segmentation_augmented \
    --preset balanced \
    --epochs 50 \
    --batch-size 8

# 6. Test on sample image
python train_segmentation.py test \
    --model checkpoints/segmentation_model_best.keras \
    --image test_images/sample.png \
    --output-dir ./test_results

# 7. Use in full pipeline
python -c "
from blood_cell_system import BloodCellAnalysisSystem
from inference_pipeline import InferencePipeline
import tensorflow as tf

system = BloodCellAnalysisSystem()
system.segmentation_model = tf.keras.models.load_model(
    'checkpoints/segmentation_model_best.keras'
)
system.load_models('./models')  # Load RBC/WBC classifiers

pipeline = InferencePipeline(system)
report = pipeline.process_image('image.png')
print(report.to_summary_text())
"
```

## 🎯 Next Steps

After training segmentation:

1. ✅ **Train RBC classifier** - See [README_SYSTEM.md](README_SYSTEM.md)
2. ✅ **Train WBC classifier** - See [README_SYSTEM.md](README_SYSTEM.md)
3. ✅ **Run end-to-end pipeline** - See [example_usage.py](example_usage.py)
4. ✅ **Batch process images** - See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

## 📚 Additional Resources

- **Data Annotation**: [CVAT](https://github.com/opencv/cvat), [LabelMe](https://github.com/wkentaro/labelme)
- **BCCD Dataset**: https://github.com/Shenggan/BCCD_Dataset
- **Kaggle Datasets**: Search "blood cell" on Kaggle
- **Paper Reference**: U-Net for biomedical segmentation

---

**Need help?** Check [README_SYSTEM.md](README_SYSTEM.md) or open an issue on GitHub.
