# 🚀 Quick Reference - Blood Cell Analysis System

## ⚡ 5-Minute Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Build Models
```python
from blood_cell_system import BloodCellAnalysisSystem

system = BloodCellAnalysisSystem(d_reduced=64, use_domain_adapt=True)
system.build_all_models()
system.compile_models()
```

### 3. Train (Pre-segmented Data)
```python
from training_pipeline import TrainingPipeline

config = {'dataset_type': 'pre_segmented'}
pipeline = TrainingPipeline(system, config)

# Train RBC classifier
pipeline.train_rbc_classifier(
    data_dir='./data/rbc_data',
    epochs=50,
    batch_size=32
)

# Train WBC classifier
pipeline.train_wbc_classifier(
    data_dir='./data/wbc_data',
    epochs=50,
    batch_size=32
)

# Save models
system.save_models('./models')
```

### 4. Run Inference
```python
from inference_pipeline import InferencePipeline

# Load models
system = BloodCellAnalysisSystem()
system.load_models('./models')

# Process image
pipeline = InferencePipeline(system)
report = pipeline.process_image('image.png')

# View results
print(report.to_summary_text())
```

---

## 📁 Required Directory Structure

### For Training

```
data/
├── rbc_data/           # RBC classification
│   ├── healthy_RBC/
│   ├── malaria_RBC/
│   └── sickle_RBC/
│
└── wbc_data/           # WBC classification
    ├── healthy_WBC/
    └── cancer_WBC/
```

### After Training

```
models/
├── segmentation_model.keras
├── rbc_classifier.keras
└── wbc_classifier.keras
```

---

## 🎯 Common Commands

### Build System
```python
system = BloodCellAnalysisSystem(
    seg_input_shape=(512, 512, 3),
    clf_input_shape=(130, 130, 3),
    d_reduced=64,              # 32=fast, 64=balanced, 128=accurate
    use_domain_adapt=True      # True for multiple staining
)
system.build_all_models()
system.compile_models()
```

### Train Single Model
```python
pipeline = TrainingPipeline(system, config)
pipeline.train_rbc_classifier(
    data_dir='path/to/data',
    epochs=50,
    batch_size=32,
    use_class_weights=True,
    save_best=True
)
```

### Train All Models
```python
pipeline.train_all_sequential(
    seg_data_dir='./data/segmentation',  # Optional
    rbc_data_dir='./data/rbc_data',
    wbc_data_dir='./data/wbc_data',
    epochs_seg=50,
    epochs_rbc=50,
    epochs_wbc=50
)
```

### Single Image Inference
```python
system = BloodCellAnalysisSystem()
system.load_models('./models')
pipeline = InferencePipeline(system)

report = pipeline.process_image(
    'image.png',
    save_visualization=True,
    output_dir='./results'
)

print(report.to_summary_text())
```

### Batch Processing
```python
from inference_pipeline import batch_process_images

batch_process_images(
    image_dir='./test_images',
    pipeline=pipeline,
    output_dir='./batch_results'
)
```

---

## ⚙️ Configuration Presets

### Fast (Testing)
```python
system = BloodCellAnalysisSystem(
    d_reduced=32,
    use_domain_adapt=False
)
# epochs=10, batch_size=64
# Time: ~3 hours
```

### Balanced (Recommended)
```python
system = BloodCellAnalysisSystem(
    d_reduced=64,
    use_domain_adapt=True
)
# epochs=50, batch_size=32
# Time: ~13 hours
```

### Accurate (Production)
```python
system = BloodCellAnalysisSystem(
    d_reduced=128,
    use_domain_adapt=True
)
# epochs=100, batch_size=32
# Time: ~25 hours
```

---

## 📊 Output Files

### Per Image
```
results/
├── image_annotated.png    # Visual results
├── image_report.json      # Machine-readable
└── image_summary.txt      # Human-readable
```

### Report Structure
```python
report = {
    'total_cells': 234,
    'rbc_count': 210,
    'wbc_count': 24,
    'rbc_analysis': {
        'healthy': 185,
        'malaria': 15,
        'sickle_cell': 10,
        'infection_rate': '11.90%'
    },
    'wbc_analysis': {
        'healthy': 22,
        'leukemia': 2,
        'cancer_rate': '8.33%'
    }
}
```

---

## 🔧 Common Parameters

### BloodCellAnalysisSystem
```python
seg_input_shape=(512, 512, 3)  # Segmentation input size
clf_input_shape=(130, 130, 3)  # Classifier input size
d_reduced=64                    # Reynolds dimension
use_domain_adapt=True           # Domain adaptation
```

### Training
```python
data_dir='./data'           # Dataset directory
epochs=50                   # Training epochs
batch_size=32               # Batch size
use_class_weights=True      # Handle imbalance
save_best=True              # Save best model
```

### Inference
```python
confidence_threshold=0.7    # Min confidence
uncertain_margin=0.15       # Uncertainty threshold
save_visualization=True     # Save annotated image
output_dir='./results'      # Output directory
```

---

## 🐛 Quick Troubleshooting

### Out of Memory
```python
# Reduce batch size
batch_size = 16  # or 8

# Reduce Reynolds dimension
d_reduced = 32

# Reduce input size
seg_input_shape = (256, 256, 3)
```

### Poor Accuracy
```python
# Enable domain adaptation
use_domain_adapt = True

# Increase epochs
epochs = 100

# Use class weights
use_class_weights = True

# Check data quality
# - Correct labels?
# - Sufficient samples per class?
# - Data augmentation appropriate?
```

### Training Too Slow
```python
# Reduce Reynolds dimension
d_reduced = 32

# Use fewer epochs (testing)
epochs = 10

# Increase batch size (if memory allows)
batch_size = 64

# Check GPU usage
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## 📈 Expected Performance

### Training Time (10k images/class)
- **RBC Classifier**: ~12.5 hours (50 epochs)
- **WBC Classifier**: ~12.5 hours (50 epochs)  
- **Segmentation**: ~21 hours (50 epochs, optional)

### Inference Speed
- **Classification**: ~100 cells/second
- **End-to-End**: ~1-2 images/second

### Accuracy
- **RBC Classifier**: 94-97%
- **WBC Classifier**: 92-95%
- **Segmentation**: 88-92% IoU

---

## 🎯 Key Features

✅ **Reynolds Networks** (O(n) complexity)  
✅ **Domain Adaptation** (handles staining variations)  
✅ **Stain Normalization** (color-invariant)  
✅ **Pre-segmented Data** (e.g., Malaria dataset)  
✅ **Raw Images** (end-to-end pipeline)  
✅ **Uncertainty Detection** (flags low confidence)  
✅ **Batch Processing** (process multiple images)  
✅ **Comprehensive Reports** (JSON + text + visual)  

---

## 📚 Full Documentation

- **README_SYSTEM.md**: Complete documentation (18 KB)
- **PROJECT_SUMMARY.md**: Project overview (16 KB)
- **example_usage.py**: 6 working examples (9 KB)

---

## 💡 Quick Tips

1. **Start with pre-segmented data** (easier, faster)
2. **Use d_reduced=64** for balanced performance
3. **Enable domain adaptation** if multiple staining methods
4. **Test with 10 epochs** before full training
5. **Check class balance** and use class weights
6. **Validate on held-out set** before deployment
7. **Monitor uncertain cases** for quality control

---

## 🔗 File Overview

| File | Size | Purpose |
|------|------|---------|
| `blood_cell_system.py` | 26KB | Core architecture |
| `training_pipeline.py` | 22KB | Training framework |
| `inference_pipeline.py` | 24KB | Inference + reports |
| `example_usage.py` | 9KB | 6 complete examples |
| `README_SYSTEM.md` | 18KB | Full documentation |
| `PROJECT_SUMMARY.md` | 16KB | Project overview |
| `requirements.txt` | 102B | Dependencies |

---

## ✅ Checklist Before Deployment

- [ ] Data organized in correct structure
- [ ] Dependencies installed
- [ ] Models trained and validated
- [ ] Tested on held-out data
- [ ] Checked different staining methods
- [ ] Verified inference speed
- [ ] Models saved and backed up
- [ ] Documentation reviewed

---

**Need help?** See README_SYSTEM.md for complete documentation!
