#!/usr/bin/env python3
"""
MSVPD Unified Dataset Merger
=============================

Merges MVTec AD and DecoSpan datasets into a unified directory structure.
- Preserves original datasets (non-destructive copy)
- Restructures DecoSpan to match MVTec format
- Generates pixel-level masks for anomaly images

Author: Data Engineering Pipeline
Date: 2026
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Optional, List
import time

# Try to import CV2, fall back to PIL if not available
try:
    import cv2
    USE_CV2 = True
except ImportError:
    try:
        from PIL import Image
        USE_CV2 = False
    except ImportError:
        print("ERROR: Neither cv2 nor PIL is available. Please install opencv-python or Pillow.")
        sys.exit(1)

import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for dataset merger."""
    
    # Source paths (from current working directory)
    MVTEC_SOURCE = "./mvtec_anomaly_detection"
    DECOSPAN_SOURCE = "./decospan_small"
    
    # Target path
    UNIFIED_TARGET = "./MSVPD_Unified_Dataset"
    
    # Subdirectories
    MVTEC_SUBDIR = "MVTec"
    DECOSPAN_SUBDIR = "Decospan"
    
    # Image parameters for mask generation
    MASK_VALUE = 255  # White masks
    MASK_FORMAT = "png"


# ============================================================================
# LOGGER - Progress Tracking
# ============================================================================

class Logger:
    """Simple logger with colored output."""
    
    COLORS = {
        "INFO": "\033[94m",      # Blue
        "SUCCESS": "\033[92m",   # Green
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "RESET": "\033[0m"       # Reset
    }
    
    @staticmethod
    def info(msg: str):
        print(f"{Logger.COLORS['INFO']}[INFO]{Logger.COLORS['RESET']} {msg}")
    
    @staticmethod
    def success(msg: str):
        print(f"{Logger.COLORS['SUCCESS']}[✓ OK]{Logger.COLORS['RESET']} {msg}")
    
    @staticmethod
    def warning(msg: str):
        print(f"{Logger.COLORS['WARNING']}[⚠ WARN]{Logger.COLORS['RESET']} {msg}")
    
    @staticmethod
    def error(msg: str):
        print(f"{Logger.COLORS['ERROR']}[✗ ERR]{Logger.COLORS['RESET']} {msg}")
    
    @staticmethod
    def header(msg: str):
        print(f"\n{'='*70}")
        print(f"  {msg}")
        print(f"{'='*70}\n")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def verify_source_exists(path: str, name: str) -> bool:
    """Verify that source dataset exists."""
    if not os.path.exists(path):
        Logger.error(f"{name} not found at: {path}")
        return False
    Logger.success(f"{name} found at: {path}")
    return True


def create_directory(path: str, description: str = "") -> bool:
    """Create directory with logging."""
    try:
        os.makedirs(path, exist_ok=True)
        Logger.info(f"Created directory: {path}" + (f" ({description})" if description else ""))
        return True
    except Exception as e:
        Logger.error(f"Failed to create directory {path}: {e}")
        return False


def copy_file(src: str, dst: str, description: str = "") -> bool:
    """Copy file with error handling."""
    try:
        shutil.copy2(src, dst)
        if description:
            Logger.info(f"Copied: {description}")
        return True
    except Exception as e:
        Logger.error(f"Failed to copy {src} to {dst}: {e}")
        return False


def copy_directory_contents(src_dir: str, dst_dir: str, pattern: str = "*") -> int:
    """
    Copy all files from src_dir to dst_dir.
    
    Args:
        src_dir: Source directory
        dst_dir: Destination directory
        pattern: File pattern to match (e.g., "*.png")
    
    Returns:
        Number of files copied
    """
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    
    if not os.path.exists(src_dir):
        Logger.warning(f"Source directory does not exist: {src_dir}")
        return count
    
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        if os.path.isfile(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                count += 1
            except Exception as e:
                Logger.error(f"Failed to copy {src_path}: {e}")
    
    return count


def get_image_dimensions(image_path: str) -> Optional[tuple]:
    """
    Get image dimensions (height, width).
    
    Args:
        image_path: Path to image file
    
    Returns:
        (height, width) tuple or None if error
    """
    try:
        if USE_CV2:
            img = cv2.imread(image_path)
            if img is None:
                Logger.error(f"cv2 failed to read: {image_path}")
                return None
            height, width = img.shape[:2]
        else:
            img = Image.open(image_path)
            width, height = img.size
        
        return (height, width)
    except Exception as e:
        Logger.error(f"Failed to get dimensions for {image_path}: {e}")
        return None


def generate_white_mask(height: int, width: int, image_path: str, output_path: str) -> bool:
    """
    Generate a white PNG mask matching image dimensions.
    
    Args:
        height: Image height
        width: Image width
        image_path: Original image path (for reference)
        output_path: Output mask path
    
    Returns:
        True if successful
    """
    try:
        # Create white image
        mask = np.full((height, width), Config.MASK_VALUE, dtype=np.uint8)
        
        # Save as PNG
        if USE_CV2:
            success = cv2.imwrite(output_path, mask)
            if not success:
                Logger.error(f"cv2.imwrite failed for: {output_path}")
                return False
        else:
            mask_img = Image.fromarray(mask, mode='L')
            mask_img.save(output_path)
        
        return True
    except Exception as e:
        Logger.error(f"Failed to generate mask for {image_path}: {e}")
        return False


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution function."""
    
    Logger.header("MSVPD Unified Dataset Merger")
    
    start_time = time.time()
    
    # ========================================================================
    # PHASE 1: VALIDATION
    # ========================================================================
    Logger.header("PHASE 1: Validating Source Datasets")
    
    mvtec_exists = verify_source_exists(Config.MVTEC_SOURCE, "MVTec AD")
    decospan_exists = verify_source_exists(Config.DECOSPAN_SOURCE, "DecoSpan Small")
    
    if not (mvtec_exists and decospan_exists):
        Logger.error("One or both source datasets are missing. Aborting.")
        sys.exit(1)
    
    # Count dataset contents
    mvtec_categories = [d for d in os.listdir(Config.MVTEC_SOURCE) 
                        if os.path.isdir(os.path.join(Config.MVTEC_SOURCE, d))]
    Logger.info(f"Found {len(mvtec_categories)} MVTec categories: {sorted(mvtec_categories)}")
    
    # ========================================================================
    # PHASE 2: CREATE TARGET STRUCTURE
    # ========================================================================
    Logger.header("PHASE 2: Creating Target Directory Structure")
    
    target_exists = os.path.exists(Config.UNIFIED_TARGET)
    if target_exists:
        Logger.warning(f"Target directory already exists: {Config.UNIFIED_TARGET}")
        response = input("Overwrite? (y/n): ").lower().strip()
        if response != 'y':
            Logger.warning("Aborting to preserve existing data.")
            sys.exit(0)
        shutil.rmtree(Config.UNIFIED_TARGET)
        Logger.info("Removed existing target directory.")
    
    # Create main directory
    create_directory(Config.UNIFIED_TARGET, "main output")
    
    # Create MVTec subdirectory
    mvtec_target = os.path.join(Config.UNIFIED_TARGET, Config.MVTEC_SUBDIR)
    create_directory(mvtec_target, "MVTec container")
    
    # Create DecoSpan subdirectories
    decospan_target = os.path.join(Config.UNIFIED_TARGET, Config.DECOSPAN_SUBDIR)
    decospan_train_good = os.path.join(decospan_target, "train", "good")
    decospan_test_good = os.path.join(decospan_target, "test", "good")
    decospan_test_custom = os.path.join(decospan_target, "test", "custom_defect")
    decospan_gt = os.path.join(decospan_target, "ground_truth", "custom_defect")
    
    for dir_path, desc in [
        (decospan_train_good, "DecoSpan train/good"),
        (decospan_test_good, "DecoSpan test/good"),
        (decospan_test_custom, "DecoSpan test/custom_defect"),
        (decospan_gt, "DecoSpan ground_truth/custom_defect"),
    ]:
        create_directory(dir_path, desc)
    
    # ========================================================================
    # PHASE 3: TRANSFER MVTec CATEGORIES
    # ========================================================================
    Logger.header("PHASE 3: Transferring MVTec AD Categories")
    
    mvtec_count = 0
    for category in sorted(mvtec_categories):
        src_cat = os.path.join(Config.MVTEC_SOURCE, category)
        dst_cat = os.path.join(mvtec_target, category)
        
        try:
            shutil.copytree(src_cat, dst_cat, dirs_exist_ok=False)
            Logger.success(f"Transferred: {category}")
            mvtec_count += 1
        except Exception as e:
            Logger.error(f"Failed to transfer {category}: {e}")
    
    Logger.info(f"Successfully transferred {mvtec_count}/{len(mvtec_categories)} categories")
    
    # ========================================================================
    # PHASE 4: RESTRUCTURE DECOSPAN
    # ========================================================================
    Logger.header("PHASE 4: Restructuring DecoSpan Dataset")
    
    # 4.1: Copy train/normal -> train/good
    Logger.info("Copying train/normal -> train/good...")
    train_normal_src = os.path.join(Config.DECOSPAN_SOURCE, "train", "normal")
    train_good_count = copy_directory_contents(train_normal_src, decospan_train_good)
    Logger.success(f"Copied {train_good_count} files: train/normal -> train/good")
    
    # 4.2: Copy test/normal -> test/good
    Logger.info("Copying test/normal -> test/good...")
    test_normal_src = os.path.join(Config.DECOSPAN_SOURCE, "test", "normal")
    test_good_count = copy_directory_contents(test_normal_src, decospan_test_good)
    Logger.success(f"Copied {test_good_count} files: test/normal -> test/good")
    
    # 4.3: Copy train/anomaly -> test/custom_defect
    Logger.info("Copying train/anomaly -> test/custom_defect...")
    train_anomaly_src = os.path.join(Config.DECOSPAN_SOURCE, "train", "anomaly")
    train_anomaly_count = copy_directory_contents(train_anomaly_src, decospan_test_custom)
    Logger.success(f"Copied {train_anomaly_count} files: train/anomaly -> test/custom_defect")
    
    # 4.4: Copy test/anomaly -> test/custom_defect
    Logger.info("Copying test/anomaly -> test/custom_defect...")
    test_anomaly_src = os.path.join(Config.DECOSPAN_SOURCE, "test", "anomaly")
    test_anomaly_count = copy_directory_contents(test_anomaly_src, decospan_test_custom)
    Logger.success(f"Copied {test_anomaly_count} files: test/anomaly -> test/custom_defect")
    
    total_anomaly = train_anomaly_count + test_anomaly_count
    Logger.info(f"Total anomaly images in test/custom_defect: {total_anomaly}")
    
    # ========================================================================
    # PHASE 5: GENERATE MASKS FOR ANOMALY IMAGES
    # ========================================================================
    Logger.header("PHASE 5: Generating Ground Truth Masks")
    
    Logger.info(f"Processing {total_anomaly} anomaly images for mask generation...")
    
    mask_count = 0
    mask_errors = 0
    
    for filename in os.listdir(decospan_test_custom):
        image_path = os.path.join(decospan_test_custom, filename)
        
        if not os.path.isfile(image_path):
            continue
        
        # Get image dimensions
        dims = get_image_dimensions(image_path)
        if dims is None:
            mask_errors += 1
            continue
        
        height, width = dims
        
        # Generate mask filename (replace extension with .png)
        mask_filename = os.path.splitext(filename)[0] + f".{Config.MASK_FORMAT}"
        mask_path = os.path.join(decospan_gt, mask_filename)
        
        # Generate mask
        if generate_white_mask(height, width, image_path, mask_path):
            mask_count += 1
            Logger.info(f"Mask {mask_count}: {mask_filename} ({height}×{width})")
        else:
            mask_errors += 1
    
    Logger.success(f"Generated {mask_count} masks ({mask_errors} errors)")
    
    # ========================================================================
    # PHASE 6: SUMMARY
    # ========================================================================
    Logger.header("PHASE 6: Summary & Verification")
    
    print(f"\nUnified Dataset Structure Created:")
    print(f"  Location: {Config.UNIFIED_TARGET}")
    print(f"\n✓ MVTec AD:")
    print(f"    Categories transferred: {mvtec_count}/{len(mvtec_categories)}")
    print(f"    Location: {mvtec_target}")
    print(f"\n✓ DecoSpan:")
    print(f"    Train (good): {train_good_count} images")
    print(f"    Test (good): {test_good_count} images")
    print(f"    Test (anomaly): {total_anomaly} images")
    print(f"    Ground truth masks: {mask_count} masks")
    print(f"    Location: {decospan_target}")
    
    # Verify directory structure
    print(f"\nDirectory Tree:")
    for root, dirs, files in os.walk(Config.UNIFIED_TARGET):
        level = root.replace(Config.UNIFIED_TARGET, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        sub_indent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files per directory
            print(f'{sub_indent}{file}')
        if len(files) > 3:
            print(f'{sub_indent}... and {len(files) - 3} more files')
    
    elapsed = time.time() - start_time
    Logger.header(f"✓ COMPLETE - Elapsed Time: {elapsed:.2f} seconds")
    
    print(f"\nNext Steps:")
    print(f"1. Verify the MSVPD_Unified_Dataset folder structure")
    print(f"2. Test loading both datasets with anomalib datamodules")
    print(f"3. Run Stage 1 (VLM) benchmarking on MVTec categories")
    print(f"4. Run Stage 3 (BGAD) supervised training on DecoSpan")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        Logger.warning("Script interrupted by user.")
        sys.exit(0)
    except Exception as e:
        Logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
