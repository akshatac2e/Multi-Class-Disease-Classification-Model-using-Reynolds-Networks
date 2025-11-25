#!/usr/bin/env python3
"""
Automated Dataset Downloader for Blood Cell Segmentation
Downloads all 4 recommended datasets: BCCD, Kaggle Blood Cell, LISC, ALL-IDB
"""

import os
import sys
import subprocess
import urllib.request
import tarfile
import zipfile
from pathlib import Path
import shutil

# Configuration
REPO_ROOT = Path("/workspaces/Multi-Class-Disease-Classification-Model-using-Reynolds-Networks")
DATA_DIR = REPO_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"

class DatasetDownloader:
    def __init__(self):
        self.success_count = 0
        self.failed_downloads = []
        
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "=" * 60)
        print(text)
        print("=" * 60 + "\n")
    
    def print_status(self, status, message):
        """Print status message"""
        symbols = {"success": "✓", "warning": "⚠", "error": "✗", "info": "ℹ"}
        print(f"{symbols.get(status, '•')} {message}")
    
    def run_command(self, cmd, cwd=None, shell=True):
        """Run shell command and return success status"""
        try:
            result = subprocess.run(
                cmd,
                shell=shell,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)
    
    def download_file(self, url, output_path):
        """Download file from URL with progress"""
        try:
            self.print_status("info", f"Downloading from {url}")
            urllib.request.urlretrieve(url, output_path)
            return True
        except Exception as e:
            self.print_status("error", f"Download failed: {e}")
            return False
    
    def extract_archive(self, archive_path, extract_to):
        """Extract tar.gz or zip archive"""
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.name.endswith('.tar.gz'):
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_to)
            self.print_status("success", f"Extracted to {extract_to}")
            return True
        except Exception as e:
            self.print_status("error", f"Extraction failed: {e}")
            return False
    
    def download_bccd(self):
        """Download BCCD Dataset (GitHub)"""
        self.print_header("[1/4] BCCD Dataset")
        
        bccd_dir = DATA_DIR / "BCCD_Dataset"
        
        if bccd_dir.exists():
            image_count = len(list(bccd_dir.rglob("*.jpg"))) + len(list(bccd_dir.rglob("*.png")))
            self.print_status("success", f"BCCD Dataset already exists ({image_count} images)")
            self.print_status("info", f"Location: {bccd_dir}")
            self.success_count += 1
            return True
        
        self.print_status("info", "Cloning BCCD Dataset from GitHub...")
        success, stdout, stderr = self.run_command(
            "git clone https://github.com/Shenggan/BCCD_Dataset.git",
            cwd=DATA_DIR
        )
        
        if success and bccd_dir.exists():
            image_count = len(list(bccd_dir.rglob("*.jpg"))) + len(list(bccd_dir.rglob("*.png")))
            self.print_status("success", f"BCCD Dataset downloaded ({image_count} images)")
            self.success_count += 1
            return True
        else:
            self.print_status("error", "BCCD Dataset download failed")
            self.failed_downloads.append("BCCD Dataset")
            return False
    
    def download_kaggle(self):
        """Download Kaggle Blood Cell Detection Dataset"""
        self.print_header("[2/4] Kaggle Blood Cell Detection Dataset")
        
        kaggle_dir = DATASETS_DIR / "kaggle_blood_cell"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if any(kaggle_dir.rglob("*.jpg")) or any(kaggle_dir.rglob("*.png")):
            image_count = len(list(kaggle_dir.rglob("*.jpg"))) + len(list(kaggle_dir.rglob("*.png")))
            self.print_status("success", f"Kaggle dataset already exists ({image_count} images)")
            self.success_count += 1
            return True
        
        # Check if kaggle CLI is installed
        success, _, _ = self.run_command("which kaggle")
        
        if not success:
            self.print_status("info", "Installing kaggle CLI...")
            success, _, _ = self.run_command("pip install -q kaggle")
            
            if not success:
                self.print_status("error", "Failed to install kaggle CLI")
                self._print_kaggle_manual_instructions()
                self.failed_downloads.append("Kaggle Blood Cell Dataset")
                return False
        
        # Check for kaggle credentials
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            self.print_status("warning", "Kaggle API credentials not found")
            self._print_kaggle_manual_instructions()
            self.failed_downloads.append("Kaggle Blood Cell Dataset (credentials required)")
            return False
        
        # Download using kaggle CLI
        self.print_status("info", "Downloading from Kaggle...")
        success, stdout, stderr = self.run_command(
            "kaggle datasets download -d dracarys/blood-cell-detection-dataset",
            cwd=kaggle_dir
        )
        
        if success:
            # Extract zip file
            zip_file = kaggle_dir / "blood-cell-detection-dataset.zip"
            if zip_file.exists():
                self.extract_archive(zip_file, kaggle_dir)
                zip_file.unlink()  # Remove zip after extraction
                
                image_count = len(list(kaggle_dir.rglob("*.jpg"))) + len(list(kaggle_dir.rglob("*.png")))
                self.print_status("success", f"Kaggle dataset downloaded ({image_count} images)")
                self.success_count += 1
                return True
        
        self.print_status("error", "Kaggle download failed")
        self._print_kaggle_manual_instructions()
        self.failed_downloads.append("Kaggle Blood Cell Dataset")
        return False
    
    def _print_kaggle_manual_instructions(self):
        """Print manual download instructions for Kaggle"""
        print("\n  Manual Download Instructions:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Create API token (downloads kaggle.json)")
        print("  3. Place at: ~/.kaggle/kaggle.json")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("  5. Re-run this script")
        print("\n  OR download manually:")
        print("  https://www.kaggle.com/datasets/dracarys/blood-cell-detection-dataset")
        print(f"  Extract to: {DATASETS_DIR}/kaggle_blood_cell/")
    
    def download_lisc(self):
        """Download LISC Dataset (Leukocyte Images)"""
        self.print_header("[3/4] LISC Dataset (Leukocyte Images)")
        
        lisc_dir = DATASETS_DIR / "LISC"
        lisc_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if any(lisc_dir.rglob("*.jpg")) or any(lisc_dir.rglob("*.png")) or any(lisc_dir.rglob("*.bmp")):
            image_count = (len(list(lisc_dir.rglob("*.jpg"))) + 
                          len(list(lisc_dir.rglob("*.png"))) + 
                          len(list(lisc_dir.rglob("*.bmp"))))
            self.print_status("success", f"LISC dataset already exists ({image_count} images)")
            self.success_count += 1
            return True
        
        # Try downloading from known URLs
        urls = [
            "https://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.zip",
            "https://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte_Data.zip"
        ]
        
        for url in urls:
            self.print_status("info", f"Attempting download from: {url}")
            zip_file = lisc_dir / "leukocyte_data.zip"
            
            if self.download_file(url, zip_file):
                if self.extract_archive(zip_file, lisc_dir):
                    zip_file.unlink()
                    image_count = (len(list(lisc_dir.rglob("*.jpg"))) + 
                                  len(list(lisc_dir.rglob("*.png"))) + 
                                  len(list(lisc_dir.rglob("*.bmp"))))
                    self.print_status("success", f"LISC dataset downloaded ({image_count} images)")
                    self.success_count += 1
                    return True
        
        self.print_status("warning", "LISC dataset download unavailable from known sources")
        print("\n  Manual Download Options:")
        print("  - Contact: https://users.cecs.anu.edu.au/~hrezatofighi/")
        print("  - Search: 'Leukocyte Images Segmentation Classification dataset'")
        print(f"  Extract to: {lisc_dir}/")
        print("\n  Note: LISC is optional. BCCD + Kaggle datasets (724 images) are sufficient.")
        
        self.failed_downloads.append("LISC Dataset (optional)")
        return False
    
    def download_allidb(self):
        """Download ALL-IDB Dataset (Acute Lymphoblastic Leukemia)"""
        self.print_header("[4/4] ALL-IDB Dataset")
        
        allidb_dir = DATASETS_DIR / "ALL-IDB"
        allidb_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if any(allidb_dir.rglob("*.jpg")) or any(allidb_dir.rglob("*.png")):
            image_count = len(list(allidb_dir.rglob("*.jpg"))) + len(list(allidb_dir.rglob("*.png")))
            self.print_status("success", f"ALL-IDB dataset already exists ({image_count} images)")
            self.success_count += 1
            return True
        
        success = True
        
        # Download ALL-IDB1 (108 images)
        allidb1_url = "https://homes.di.unimi.it/scotti/all/ALL_IDB1.tar.gz"
        allidb1_tar = allidb_dir / "ALL_IDB1.tar.gz"
        
        self.print_status("info", "Downloading ALL-IDB1 (108 images)...")
        if self.download_file(allidb1_url, allidb1_tar):
            if self.extract_archive(allidb1_tar, allidb_dir):
                allidb1_tar.unlink()
                self.print_status("success", "ALL-IDB1 downloaded and extracted")
            else:
                success = False
        else:
            self.print_status("warning", "ALL-IDB1 download failed")
            success = False
        
        # Download ALL-IDB2 (260 images)
        allidb2_url = "https://homes.di.unimi.it/scotti/all/ALL_IDB2.tar.gz"
        allidb2_tar = allidb_dir / "ALL_IDB2.tar.gz"
        
        self.print_status("info", "Downloading ALL-IDB2 (260 images)...")
        if self.download_file(allidb2_url, allidb2_tar):
            if self.extract_archive(allidb2_tar, allidb_dir):
                allidb2_tar.unlink()
                self.print_status("success", "ALL-IDB2 downloaded and extracted")
            else:
                success = False
        else:
            self.print_status("warning", "ALL-IDB2 download failed")
            success = False
        
        if success:
            image_count = len(list(allidb_dir.rglob("*.jpg"))) + len(list(allidb_dir.rglob("*.png")))
            self.print_status("success", f"ALL-IDB dataset complete ({image_count} images)")
            self.success_count += 1
            return True
        else:
            self.print_status("warning", "Some ALL-IDB files failed to download")
            print("\n  Manual Download:")
            print("  1. Visit: https://homes.di.unimi.it/scotti/all/")
            print("  2. Download ALL_IDB1.tar.gz and ALL_IDB2.tar.gz")
            print(f"  3. Extract to: {allidb_dir}/")
            
            self.failed_downloads.append("ALL-IDB Dataset (partial)")
            return False
    
    def print_summary(self):
        """Print download summary and next steps"""
        self.print_header("Download Summary")
        
        # Count images in each dataset
        datasets = {
            "BCCD Dataset": DATA_DIR / "BCCD_Dataset",
            "Kaggle Blood Cell": DATASETS_DIR / "kaggle_blood_cell",
            "LISC Dataset": DATASETS_DIR / "LISC",
            "ALL-IDB Dataset": DATASETS_DIR / "ALL-IDB"
        }
        
        total_images = 0
        for name, path in datasets.items():
            if path.exists():
                image_count = (len(list(path.rglob("*.jpg"))) + 
                              len(list(path.rglob("*.png"))) + 
                              len(list(path.rglob("*.bmp"))))
                if image_count > 0:
                    self.print_status("success", f"{name}: {image_count} images")
                    total_images += image_count
                else:
                    self.print_status("warning", f"{name}: Not downloaded")
            else:
                self.print_status("warning", f"{name}: Not found")
        
        print(f"\nTotal images downloaded: {total_images}")
        
        if self.failed_downloads:
            print(f"\nFailed/Incomplete downloads ({len(self.failed_downloads)}):")
            for item in self.failed_downloads:
                print(f"  • {item}")
        
        self.print_header("Next Steps")
        
        print("1. Prepare segmentation data from BCCD Dataset:")
        print("   python prepare_segmentation_data.py \\")
        print("     --bccd-dir data/BCCD_Dataset \\")
        print("     --output-dir data/segmentation_prepared \\")
        print("     --augmentation-factor 3")
        print()
        print("2. Verify prepared data:")
        print("   python prepare_segmentation_data.py \\")
        print("     --verify data/segmentation_prepared")
        print()
        print("3. Train segmentation model:")
        print("   python train_segmentation.py train \\")
        print("     --data-dir data/segmentation_prepared \\")
        print("     --preset balanced \\")
        print("     --epochs 50")
        print()
        print("4. Test segmentation model:")
        print("   python train_segmentation.py test \\")
        print("     --model-path models/segmentation_model.h5 \\")
        print("     --image data/BCCD_Dataset/BCCD/JPEGImages/BloodImage_00001.jpg")
        print()
        print("For detailed instructions, see:")
        print("  • DATASET_DOWNLOAD_INSTRUCTIONS.md")
        print("  • SEGMENTATION_GUIDE.md")
        print("  • README_SYSTEM.md")
        
    def run(self):
        """Main execution function"""
        self.print_header("Blood Cell Dataset Downloader")
        print("Downloading 4 recommended datasets for segmentation training...")
        
        # Create directory structure
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download each dataset
        self.download_bccd()
        self.download_kaggle()
        self.download_lisc()
        self.download_allidb()
        
        # Print summary
        self.print_summary()
        
        return self.success_count >= 2  # Success if at least 2 datasets downloaded


def main():
    """Entry point"""
    try:
        downloader = DatasetDownloader()
        success = downloader.run()
        
        if success:
            print("\n✓ Dataset download process completed!")
            sys.exit(0)
        else:
            print("\n⚠ Some downloads failed. Check instructions above for manual download.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n✗ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
