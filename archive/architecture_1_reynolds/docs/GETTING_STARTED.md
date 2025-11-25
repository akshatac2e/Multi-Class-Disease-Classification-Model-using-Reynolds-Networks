# 🚀 Getting Started Guide

This guide will walk you through setting up and using the Blood Cell Analysis System.

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- GPU recommended but not required

## 🔧 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/akshatac2e/Multi-Class-Disease-Classification-Model-using-Reynolds-Networks.git
cd Multi-Class-Disease-Classification-Model-using-Reynolds-Networks
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Linux/Mac
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Setup

```bash
python setup.py
```

You should see all checks passing. If not, review the error messages.

## 📊 Preparing Your Data

### Option 1: Pre-segmented Dataset (Easiest)

If you have single-cell images (like the Malaria dataset):

```
data/
├── rbc_data/
│   ├── healthy_RBC/
│   │   ├── cell001.png
│   │   ├── cell002.png
│   │   └── ...
│   ├── malaria_RBC/
│   │   └── ...
│   └── sickle_RBC/
│       └── ...
└── wbc_data/
    ├── healthy_WBC/
    │   └── ...
    └── cancer_WBC/
        └── ...
```

### Option 2: Raw Microscopy Images

If you have full microscopy images with multiple cells, you'll also need:

```
data/
└── segmentation/
    ├── images/
    │   ├── img001.png
    │   └── ...
    └── masks/
        ├── img001.png  # Corresponding mask
        └── ...
```

**Mask format**: RGB image where:
- Red channel (255) = Background
- Green channel (255) = RBC
- Blue channel (255) = WBC

## 🏃 Quick Start Examples

### Example 1: Build and Inspect Models

```bash
python example_usage.py
```

This will build all three models and show their architectures.

### Example 2: Train RBC Classifier Only

```python
from blood_cell_system import BloodCellAnalysisSystem
from training_pipeline import TrainingPipeline

# Build system
system = BloodCellAnalysisSystem(
    d_reduced=64,
    use_domain_adapt=True
)
system.build_all_models()
system.compile_models()

# Train RBC classifier
config = {'dataset_type': 'pre_segmented'}
pipeline = TrainingPipeline(system, config)

pipeline.train_rbc_classifier(
    data_dir='./data/rbc_data',
    epochs=50,
    batch_size=32
)

# Save model
system.save_models('./models')
```

### Example 3: Run Inference

```python
from blood_cell_system import BloodCellAnalysisSystem
from inference_pipeline import InferencePipeline

# Load trained models
system = BloodCellAnalysisSystem()
system.load_models('./models')

# Create inference pipeline
pipeline = InferencePipeline(system)

# Process image
report = pipeline.process_image(
    'test_image.png',
    save_visualization=True,
    output_dir='./results'
)

# Print summary
print(report.to_summary_text())
```

## 🎓 Training Your First Model

### Step-by-Step Training

1. **Prepare your data** (see Data Preparation above)

2. **Choose a configuration preset**:
   - **Fast** (3 hours): For testing
   - **Balanced** (13 hours): Recommended
   - **Accurate** (25 hours): For production

3. **Run training**:

```python
from blood_cell_system import BloodCellAnalysisSystem
from training_pipeline import TrainingPipeline

# Balanced preset
system = BloodCellAnalysisSystem(
    d_reduced=64,
    use_domain_adapt=True
)
system.build_all_models()
system.compile_models()

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

4. **Monitor training**:
   - Watch the output for accuracy and loss
   - Check `checkpoints/` for saved checkpoints
   - Review `training_history.json` after training

## 🔬 Running Inference

### Single Image

```python
from inference_pipeline import InferencePipeline
from blood_cell_system import BloodCellAnalysisSystem

# Load models
system = BloodCellAnalysisSystem()
system.load_models('./models')

# Create pipeline
pipeline = InferencePipeline(system)

# Process
report = pipeline.process_image('image.png', output_dir='./results')
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

## 📈 Understanding the Output

After inference, you'll get three files per image:

1. **`image_annotated.png`**: Visual results with bounding boxes
2. **`image_report.json`**: Complete machine-readable report
3. **`image_summary.txt`**: Human-readable summary

Example summary:
```
================================================================================
BLOOD CELL ANALYSIS REPORT
================================================================================
Total Cells Detected: 234
  - Red Blood Cells (RBC): 210
  - White Blood Cells (WBC): 24

RED BLOOD CELL ANALYSIS:
Healthy RBCs: 185 (88.1%)
Malaria-infected RBCs: 15 (7.1%)
Sickle Cell RBCs: 10 (4.8%)
Overall RBC Infection Rate: 11.90%
```

## 🐛 Troubleshooting

### Out of Memory

```python
# Reduce batch size
batch_size = 16  # or 8

# Reduce Reynolds dimension
d_reduced = 32

# Reduce input size
seg_input_shape = (256, 256, 3)
```

### Training Too Slow

```python
# Use Fast preset
system = BloodCellAnalysisSystem(d_reduced=32, use_domain_adapt=False)
epochs = 10

# Or check GPU
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
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
# - Are labels correct?
# - Sufficient samples per class?
# - Images properly formatted?
```

## 📚 Next Steps

1. **Read the full documentation**: See [README_SYSTEM.md](README_SYSTEM.md)
2. **Try the examples**: Run all 6 examples in `example_usage.py`
3. **Experiment with configurations**: Try different presets
4. **Review the code**: Understand the architecture in `blood_cell_system.py`
5. **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## 🆘 Getting Help

- **Documentation**: Check README_SYSTEM.md, QUICK_REFERENCE.md
- **Examples**: See example_usage.py
- **Issues**: Open an issue on GitHub
- **Discussions**: Start a discussion for questions

## ⚠️ Important Notes

1. **Not for clinical use**: This is for research/educational purposes
2. **Data quality matters**: Garbage in, garbage out
3. **Start small**: Test with small datasets first
4. **GPU recommended**: Training is much faster with GPU
5. **Backup your models**: Save checkpoints regularly

## ✅ Checklist

Before deploying:
- [ ] Data properly organized
- [ ] Models trained and validated
- [ ] Tested on held-out data
- [ ] Verified different staining methods
- [ ] Checked inference speed
- [ ] Models saved and backed up

---

**Happy analyzing! 🔬🩸**

Need more help? Check the [complete documentation](README_SYSTEM.md) or open an issue on GitHub.
