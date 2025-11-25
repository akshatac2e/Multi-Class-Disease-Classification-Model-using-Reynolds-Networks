#!/bin/bash
# Quick Setup Script for Segmentation Training
# ============================================

set -e  # Exit on error

echo "============================================"
echo "Blood Cell Segmentation - Quick Setup"
echo "============================================"
echo ""

# Check if data directory exists
if [ ! -d "data/segmentation" ]; then
    echo "📂 Creating data directories..."
    mkdir -p data/segmentation/images
    mkdir -p data/segmentation/masks
    echo "✓ Directories created"
    echo ""
fi

# Function to download BCCD dataset
download_bccd() {
    echo "📥 Downloading BCCD Dataset..."
    
    if [ -d "BCCD_Dataset" ]; then
        echo "⚠️  BCCD_Dataset already exists. Skipping download."
    else
        git clone https://github.com/Shenggan/BCCD_Dataset.git
        echo "✓ BCCD Dataset downloaded"
    fi
    echo ""
}

# Function to process BCCD dataset
process_bccd() {
    echo "🔄 Processing BCCD Dataset..."
    
    if [ ! -d "BCCD_Dataset" ]; then
        echo "❌ BCCD_Dataset not found. Run: $0 download"
        exit 1
    fi
    
    python prepare_segmentation_data.py \
        --mode bccd \
        --input ./BCCD_Dataset \
        --output ./data/segmentation
    
    echo "✓ Dataset processed"
    echo ""
}

# Function to augment dataset
augment_data() {
    echo "🎨 Augmenting dataset (3× increase)..."
    
    if [ ! -d "data/segmentation/images" ]; then
        echo "❌ Segmentation data not found. Run: $0 process"
        exit 1
    fi
    
    # Check if augmented already exists
    if [ -d "data/segmentation_augmented" ]; then
        echo "⚠️  Augmented dataset exists. Remove it first if you want to regenerate."
        read -p "Remove and regenerate? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf data/segmentation_augmented
        else
            echo "Skipping augmentation."
            return
        fi
    fi
    
    python prepare_segmentation_data.py \
        --mode augment \
        --input ./data/segmentation \
        --output ./data/segmentation_augmented \
        --aug-factor 3
    
    echo "✓ Dataset augmented"
    echo ""
}

# Function to verify dataset
verify_data() {
    echo "✅ Verifying dataset..."
    
    if [ -d "data/segmentation_augmented" ]; then
        python prepare_segmentation_data.py \
            --mode verify \
            --input ./data/segmentation_augmented
    elif [ -d "data/segmentation" ]; then
        python prepare_segmentation_data.py \
            --mode verify \
            --input ./data/segmentation
    else
        echo "❌ No dataset found. Run: $0 setup"
        exit 1
    fi
    echo ""
}

# Function to train model
train_model() {
    local preset=${1:-balanced}
    local epochs=${2:-50}
    
    echo "🚀 Training segmentation model..."
    echo "   Preset: $preset"
    echo "   Epochs: $epochs"
    echo ""
    
    # Use augmented if available, otherwise use regular
    local data_dir="./data/segmentation_augmented"
    if [ ! -d "$data_dir" ]; then
        data_dir="./data/segmentation"
    fi
    
    if [ ! -d "$data_dir" ]; then
        echo "❌ No dataset found. Run: $0 setup"
        exit 1
    fi
    
    echo "   Using data from: $data_dir"
    echo ""
    
    python train_segmentation.py train \
        --data-dir "$data_dir" \
        --preset "$preset" \
        --epochs "$epochs" \
        --batch-size 8
    
    echo ""
    echo "✓ Training complete!"
    echo ""
}

# Function to test model
test_model() {
    local image_path=${1:-test_images/sample.png}
    
    echo "🔬 Testing segmentation model..."
    
    if [ ! -f "checkpoints/segmentation_model_best.keras" ]; then
        echo "❌ Trained model not found. Run: $0 train"
        exit 1
    fi
    
    if [ ! -f "$image_path" ]; then
        echo "❌ Test image not found: $image_path"
        exit 1
    fi
    
    python train_segmentation.py test \
        --model checkpoints/segmentation_model_best.keras \
        --image "$image_path" \
        --output-dir ./test_results
    
    echo "✓ Test complete! Check ./test_results/"
    echo ""
}

# Function to do full setup
full_setup() {
    echo "🎯 Running full setup..."
    echo ""
    
    download_bccd
    process_bccd
    augment_data
    verify_data
    
    echo "============================================"
    echo "✓ Setup complete!"
    echo "============================================"
    echo ""
    echo "Next steps:"
    echo "  1. Train model: $0 train"
    echo "  2. Or train fast: $0 train fast 10"
    echo "  3. Test model: $0 test path/to/image.png"
    echo ""
}

# Main script
case "${1:-help}" in
    download)
        download_bccd
        ;;
    process)
        process_bccd
        ;;
    augment)
        augment_data
        ;;
    verify)
        verify_data
        ;;
    train)
        train_model "${2:-balanced}" "${3:-50}"
        ;;
    test)
        test_model "${2}"
        ;;
    setup)
        full_setup
        ;;
    help|*)
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  setup              - Complete setup (download, process, augment)"
        echo "  download           - Download BCCD dataset"
        echo "  process            - Process BCCD to segmentation format"
        echo "  augment            - Augment dataset (3× increase)"
        echo "  verify             - Verify dataset structure"
        echo "  train [preset] [epochs]  - Train model"
        echo "                       Presets: fast, balanced, accurate"
        echo "                       Default: balanced 50"
        echo "  test [image]       - Test trained model on image"
        echo "  help               - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 setup                    # Full setup"
        echo "  $0 train balanced 50        # Train 50 epochs"
        echo "  $0 train fast 10            # Quick test"
        echo "  $0 test sample.png          # Test on image"
        echo ""
        echo "Quick workflow:"
        echo "  1. $0 setup"
        echo "  2. $0 train"
        echo "  3. $0 test test_images/sample.png"
        echo ""
        ;;
esac
