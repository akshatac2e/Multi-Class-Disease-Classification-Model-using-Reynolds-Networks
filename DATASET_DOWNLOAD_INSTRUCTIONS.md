# Dataset Download Instructions

This guide provides step-by-step instructions for downloading all 4 recommended datasets for blood cell segmentation training.

## Overview

| Dataset | Images | Type | Size | Download Method |
|---------|--------|------|------|-----------------|
| **BCCD Dataset** | 360 | Blood cells | ~7.4 MB | Git clone (✓ Already downloaded) |
| **Kaggle Blood Cell** | 364 | Blood cells | ~50 MB | Kaggle API or manual |
| **LISC** | 400+ | Leukocytes | ~100 MB | Manual download |
| **ALL-IDB** | 368 | Leukemia cells | ~200 MB | Direct download |

---

## 1. BCCD Dataset ✓ (Already Downloaded)

**Status:** ✓ Already cloned to `data/BCCD_Dataset/`

**Verification:**
```bash
ls -la data/BCCD_Dataset/
```

You should see:
- `BCCD/`: Main directory with 360 images
- `Annotations/`: XML files with bounding box annotations
- `ImageSets/`: Train/test splits

**Next Step:** This is your primary dataset. Proceed to data preparation:
```bash
python prepare_segmentation_data.py \
  --bccd-dir data/BCCD_Dataset \
  --output-dir data/segmentation_prepared
```

---

## 2. Kaggle Blood Cell Detection Dataset

**Download Method 1: Kaggle CLI (Recommended)**

### Step 1: Install Kaggle CLI
```bash
pip install kaggle
```

### Step 2: Setup Kaggle API Credentials
1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token" (downloads `kaggle.json`)
4. Move the file:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Download Dataset
```bash
cd data
mkdir -p kaggle_blood_cell
cd kaggle_blood_cell
kaggle datasets download -d dracarys/blood-cell-detection-dataset
unzip blood-cell-detection-dataset.zip
rm blood-cell-detection-dataset.zip
```

**Download Method 2: Manual Download**

1. Visit: https://www.kaggle.com/datasets/dracarys/blood-cell-detection-dataset
2. Click "Download" button (requires Kaggle account)
3. Extract the zip file to: `data/kaggle_blood_cell/`

**Verification:**
```bash
find data/kaggle_blood_cell -name "*.jpg" | wc -l
# Should show: 364
```

---

## 3. LISC Dataset (Leukocyte Images)

**Download Method: Manual Download Required**

The LISC (Leukocyte Images for Segmentation and Classification) dataset is hosted on institutional servers.

### Option 1: Direct Download (if available)
```bash
cd data
mkdir -p LISC
cd LISC
wget "https://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.zip"
unzip "Leukocyte Data.zip"
```

### Option 2: Manual Download
1. Visit: https://users.cecs.anu.edu.au/~hrezatofighi/
2. Look for "Leukocyte Data" download link
3. Extract to: `data/LISC/`

### Option 3: Alternative Sources
- Search for: "Leukocyte Images Segmentation Classification dataset"
- Check research paper repositories
- Contact: hrezatofighi at anu.edu.au

**Expected Structure:**
```
data/LISC/
├── images/       # Raw microscopy images
├── masks/        # Ground truth segmentation masks
└── annotations/  # Metadata
```

**Verification:**
```bash
find data/LISC -name "*.jpg" -o -name "*.png" -o -name "*.bmp" | wc -l
# Should show: 400+
```

---

## 4. ALL-IDB Dataset (Acute Lymphoblastic Leukemia)

**Download Method: Direct Download from Official Source**

The ALL-IDB dataset is publicly available from the University of Milan.

### Download ALL-IDB1 (108 images)
```bash
cd data
mkdir -p ALL-IDB
cd ALL-IDB
wget https://homes.di.unimi.it/scotti/all/ALL_IDB1.tar.gz
tar -xzf ALL_IDB1.tar.gz
rm ALL_IDB1.tar.gz
```

### Download ALL-IDB2 (260 images)
```bash
cd data/ALL-IDB
wget https://homes.di.unimi.it/scotti/all/ALL_IDB2.tar.gz
tar -xzf ALL_IDB2.tar.gz
rm ALL_IDB2.tar.gz
```

**Alternative: Manual Download**
1. Visit: https://homes.di.unimi.it/scotti/all/
2. Download both:
   - `ALL_IDB1.tar.gz` (108 images, ~90 MB)
   - `ALL_IDB2.tar.gz` (260 images, ~120 MB)
3. Extract to: `data/ALL-IDB/`

**Expected Structure:**
```
data/ALL-IDB/
├── ALL_IDB1/
│   ├── Im001_1.jpg
│   ├── Im002_0.jpg
│   └── ...
└── ALL_IDB2/
    ├── Im001_1.jpg
    ├── Im002_0.jpg
    └── ...
```

**Verification:**
```bash
find data/ALL-IDB -name "*.jpg" | wc -l
# Should show: 368 (108 + 260)
```

---

## Complete Download Script

For convenience, you can use the provided `download_datasets.sh` script:

```bash
chmod +x download_datasets.sh
./download_datasets.sh
```

This script will:
1. ✓ Verify BCCD Dataset (already downloaded)
2. Attempt Kaggle download (if credentials configured)
3. Attempt LISC download (may require manual intervention)
4. Download ALL-IDB1 and ALL-IDB2
5. Provide summary and next steps

---

## Dataset Summary After Download

Once all datasets are downloaded, your directory structure should look like:

```
data/
├── BCCD_Dataset/          # ✓ 360 images (Primary dataset)
│   ├── BCCD/
│   ├── Annotations/
│   └── ImageSets/
├── kaggle_blood_cell/     # 364 images
│   ├── images/
│   └── annotations/
├── LISC/                  # 400+ images
│   ├── images/
│   └── masks/
├── ALL-IDB/               # 368 images
│   ├── ALL_IDB1/         # 108 images
│   └── ALL_IDB2/         # 260 images
├── rbc_data/              # For RBC classifier training
├── wbc_data/              # For WBC classifier training
└── segmentation/          # For prepared segmentation data
```

**Total:** ~1,492 raw images available for training

---

## Next Steps After Download

### 1. Prepare Segmentation Data (BCCD - Primary)
```bash
python prepare_segmentation_data.py \
  --bccd-dir data/BCCD_Dataset \
  --output-dir data/segmentation_prepared \
  --augmentation-factor 3
```

This will create 1,080 training images (360 × 3).

### 2. Verify Prepared Data
```bash
python prepare_segmentation_data.py \
  --verify data/segmentation_prepared
```

### 3. Train Segmentation Model
```bash
python train_segmentation.py train \
  --data-dir data/segmentation_prepared \
  --preset balanced \
  --epochs 50 \
  --batch-size 8
```

Expected training time: 2-3 hours
Expected IoU: 0.80-0.88

### 4. Test Segmentation Model
```bash
python train_segmentation.py test \
  --model-path models/segmentation_model.h5 \
  --image data/BCCD_Dataset/BCCD/JPEGImages/BloodImage_00001.jpg \
  --output results/test_segmentation.png
```

---

## Troubleshooting

### Kaggle Download Fails
**Error:** `401 - Unauthorized`
**Solution:** Reconfigure Kaggle API credentials (see Section 2 above)

### LISC Download Unavailable
**Solution:** The LISC dataset may have restricted access. Use BCCD and Kaggle datasets instead (724 images combined is sufficient).

### ALL-IDB Download Slow
**Solution:** The University of Milan servers may be slow. Try using `curl` instead of `wget`:
```bash
curl -O https://homes.di.unimi.it/scotti/all/ALL_IDB1.tar.gz
curl -O https://homes.di.unimi.it/scotti/all/ALL_IDB2.tar.gz
```

### Insufficient Disk Space
Check available space:
```bash
df -h .
```

Total space needed: ~400 MB for raw datasets + ~1 GB for prepared/augmented data

---

## Dataset Licensing

- **BCCD Dataset:** MIT License (free for research and commercial use)
- **Kaggle Blood Cell:** CC0: Public Domain
- **LISC:** Research use (check with authors for commercial use)
- **ALL-IDB:** Free for research use (cite the original paper)

**Citation for ALL-IDB:**
```
Labati, R. D., Piuri, V., & Scotti, F. (2011).
ALL-IDB: The acute lymphoblastic leukemia image database for image processing.
In 2011 18th IEEE International Conference on Image Processing (pp. 2045-2048).
```

---

## Quick Reference Commands

```bash
# Download all datasets (automated)
./download_datasets.sh

# Verify BCCD (already downloaded)
ls -la data/BCCD_Dataset/

# Download Kaggle dataset
kaggle datasets download -d dracarys/blood-cell-detection-dataset

# Download ALL-IDB datasets
wget https://homes.di.unimi.it/scotti/all/ALL_IDB1.tar.gz
wget https://homes.di.unimi.it/scotti/all/ALL_IDB2.tar.gz

# Count total images
find data/ -name "*.jpg" -o -name "*.png" | wc -l

# Prepare segmentation data
python prepare_segmentation_data.py --bccd-dir data/BCCD_Dataset --output-dir data/segmentation_prepared

# Train segmentation model
python train_segmentation.py train --data-dir data/segmentation_prepared --preset balanced --epochs 50
```

---

For more information, see:
- `SEGMENTATION_GUIDE.md` - Complete segmentation training guide
- `README_SYSTEM.md` - Technical system documentation
- `QUICK_REFERENCE.md` - Command reference
