#!/bin/bash

# Script to download all 4 recommended datasets for blood cell segmentation
# This script downloads BCCD, Kaggle Blood Cell Detection, LISC, and ALL-IDB datasets

set -e  # Exit on error

REPO_ROOT="/workspaces/Multi-Class-Disease-Classification-Model-using-Reynolds-Networks"
DATA_DIR="${REPO_ROOT}/data"

echo "==========================================="
echo "Blood Cell Dataset Downloader"
echo "==========================================="
echo ""

# Create data directory structure
mkdir -p "${DATA_DIR}/datasets"
cd "${DATA_DIR}/datasets"

# ============================================
# 1. BCCD Dataset (Primary - Already Downloaded)
# ============================================
echo "[1/4] BCCD Dataset status..."
if [ -d "${DATA_DIR}/BCCD_Dataset" ]; then
    echo "✓ BCCD Dataset already downloaded (360 images)"
    echo "  Location: ${DATA_DIR}/BCCD_Dataset"
else
    echo "Downloading BCCD Dataset from GitHub..."
    cd "${DATA_DIR}"
    git clone https://github.com/Shenggan/BCCD_Dataset.git
    echo "✓ BCCD Dataset downloaded successfully"
fi
echo ""

# ============================================
# 2. Kaggle Blood Cell Detection Dataset
# ============================================
echo "[2/4] Kaggle Blood Cell Detection Dataset..."
mkdir -p "${DATA_DIR}/datasets/kaggle_blood_cell"
cd "${DATA_DIR}/datasets/kaggle_blood_cell"

# Check if kaggle CLI is available
if command -v kaggle &> /dev/null; then
    echo "Found kaggle CLI, attempting download..."
    if kaggle datasets download -d dracarys/blood-cell-detection-dataset; then
        echo "Extracting dataset..."
        unzip -q blood-cell-detection-dataset.zip
        rm blood-cell-detection-dataset.zip
        echo "✓ Kaggle Blood Cell Detection Dataset downloaded (364 images)"
    else
        echo "⚠ Kaggle download failed. Please download manually from:"
        echo "  https://www.kaggle.com/datasets/dracarys/blood-cell-detection-dataset"
        echo "  Extract to: ${DATA_DIR}/datasets/kaggle_blood_cell/"
    fi
else
    echo "⚠ Kaggle CLI not installed. Installing now..."
    pip install -q kaggle
    echo ""
    echo "⚠ Kaggle API credentials required!"
    echo "  1. Go to https://www.kaggle.com/account"
    echo "  2. Create API token (downloads kaggle.json)"
    echo "  3. Place at: ~/.kaggle/kaggle.json"
    echo "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo "  5. Then re-run this script"
    echo ""
    echo "  OR download manually from:"
    echo "  https://www.kaggle.com/datasets/dracarys/blood-cell-detection-dataset"
    echo "  Extract to: ${DATA_DIR}/datasets/kaggle_blood_cell/"
fi
echo ""

# ============================================
# 3. LISC Dataset (Leukocyte Images for Segmentation)
# ============================================
echo "[3/4] LISC Dataset (Leukocyte Images)..."
mkdir -p "${DATA_DIR}/datasets/LISC"
cd "${DATA_DIR}/datasets/LISC"

echo "Attempting to download LISC dataset..."
# LISC is typically hosted on institutional servers, trying common sources
if wget -q --spider "https://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.zip"; then
    wget -O leukocyte_data.zip "https://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.zip"
    unzip -q leukocyte_data.zip
    rm leukocyte_data.zip
    echo "✓ LISC Dataset downloaded (400+ images)"
else
    echo "⚠ LISC Dataset direct download unavailable"
    echo "  Manual download options:"
    echo "  - Contact: https://users.cecs.anu.edu.au/~hrezatofighi/"
    echo "  - Alternative: Search for 'Leukocyte Images Segmentation Classification dataset'"
    echo "  Extract to: ${DATA_DIR}/datasets/LISC/"
fi
echo ""

# ============================================
# 4. ALL-IDB Dataset (Acute Lymphoblastic Leukemia)
# ============================================
echo "[4/4] ALL-IDB Dataset..."
mkdir -p "${DATA_DIR}/datasets/ALL-IDB"
cd "${DATA_DIR}/datasets/ALL-IDB"

echo "Downloading ALL-IDB datasets from official source..."

# ALL-IDB1 (108 images)
if [ ! -f "ALL_IDB1.tar.gz" ]; then
    echo "Downloading ALL-IDB1..."
    wget -q https://homes.di.unimi.it/scotti/all/ALL_IDB1.tar.gz || \
    curl -s -O https://homes.di.unimi.it/scotti/all/ALL_IDB1.tar.gz || \
    echo "⚠ ALL-IDB1 download failed"
    
    if [ -f "ALL_IDB1.tar.gz" ]; then
        tar -xzf ALL_IDB1.tar.gz
        rm ALL_IDB1.tar.gz
        echo "✓ ALL-IDB1 extracted (108 images)"
    fi
fi

# ALL-IDB2 (260 images)
if [ ! -f "ALL_IDB2.tar.gz" ]; then
    echo "Downloading ALL-IDB2..."
    wget -q https://homes.di.unimi.it/scotti/all/ALL_IDB2.tar.gz || \
    curl -s -O https://homes.di.unimi.it/scotti/all/ALL_IDB2.tar.gz || \
    echo "⚠ ALL-IDB2 download failed"
    
    if [ -f "ALL_IDB2.tar.gz" ]; then
        tar -xzf ALL_IDB2.tar.gz
        rm ALL_IDB2.tar.gz
        echo "✓ ALL-IDB2 extracted (260 images)"
    fi
fi

if [ ! -d "ALL_IDB1" ] && [ ! -d "ALL_IDB2" ]; then
    echo "⚠ ALL-IDB download failed from primary source"
    echo "  Manual download:"
    echo "  1. Visit: https://homes.di.unimi.it/scotti/all/"
    echo "  2. Download ALL_IDB1.tar.gz and ALL_IDB2.tar.gz"
    echo "  3. Extract to: ${DATA_DIR}/datasets/ALL-IDB/"
fi
echo ""

# ============================================
# Summary
# ============================================
echo "==========================================="
echo "Dataset Download Summary"
echo "==========================================="
echo ""

total_count=0
if [ -d "${DATA_DIR}/BCCD_Dataset" ]; then
    bccd_count=$(find "${DATA_DIR}/BCCD_Dataset" -name "*.jpg" -o -name "*.png" | wc -l)
    echo "✓ BCCD Dataset: ${bccd_count} images"
    total_count=$((total_count + bccd_count))
fi

if [ -d "${DATA_DIR}/datasets/kaggle_blood_cell" ]; then
    kaggle_count=$(find "${DATA_DIR}/datasets/kaggle_blood_cell" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
    if [ "$kaggle_count" -gt 0 ]; then
        echo "✓ Kaggle Blood Cell: ${kaggle_count} images"
        total_count=$((total_count + kaggle_count))
    else
        echo "⚠ Kaggle Blood Cell: Not downloaded (manual download required)"
    fi
fi

if [ -d "${DATA_DIR}/datasets/LISC" ]; then
    lisc_count=$(find "${DATA_DIR}/datasets/LISC" -name "*.jpg" -o -name "*.png" -o -name "*.bmp" 2>/dev/null | wc -l)
    if [ "$lisc_count" -gt 0 ]; then
        echo "✓ LISC Dataset: ${lisc_count} images"
        total_count=$((total_count + lisc_count))
    else
        echo "⚠ LISC Dataset: Not downloaded (manual download required)"
    fi
fi

if [ -d "${DATA_DIR}/datasets/ALL-IDB" ]; then
    allidb_count=$(find "${DATA_DIR}/datasets/ALL-IDB" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
    if [ "$allidb_count" -gt 0 ]; then
        echo "✓ ALL-IDB Dataset: ${allidb_count} images"
        total_count=$((total_count + allidb_count))
    else
        echo "⚠ ALL-IDB Dataset: Not downloaded (manual download required)"
    fi
fi

echo ""
echo "Total images downloaded: ${total_count}"
echo ""
echo "==========================================="
echo "Next Steps:"
echo "==========================================="
echo "1. For successful downloads, prepare segmentation data:"
echo "   python prepare_segmentation_data.py --help"
echo ""
echo "2. For manual downloads, visit the URLs shown above"
echo "   and extract to the specified directories"
echo ""
echo "3. Start with BCCD Dataset (already downloaded):"
echo "   python prepare_segmentation_data.py \\"
echo "     --bccd-dir data/BCCD_Dataset \\"
echo "     --output-dir data/segmentation_prepared"
echo ""
echo "4. Train segmentation model:"
echo "   python train_segmentation.py train \\"
echo "     --data-dir data/segmentation_prepared \\"
echo "     --preset balanced \\"
echo "     --epochs 50"
echo ""
echo "==========================================="
