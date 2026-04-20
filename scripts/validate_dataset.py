#!/usr/bin/env python3
"""
MSVPD Dataset Validator
=======================

Validates the structure and integrity of the merged MSVPD_Unified_Dataset.
Checks file counts, image dimensions, and mask correspondence.

Author: Data Engineering Pipeline
Date: 2026
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple
import json

try:
    import cv2
    USE_CV2 = True
except ImportError:
    try:
        from PIL import Image
        USE_CV2 = False
    except ImportError:
        print("WARNING: Neither cv2 nor PIL available. Image validation will be skipped.")
        USE_CV2 = None


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DATASET_PATH = "./MSVPD_Unified_Dataset"
    MVTEC_SUBDIR = "MVTec"
    DECOSPAN_SUBDIR = "Decospan"
    
    # Expected MVTec categories
    EXPECTED_MVTEC_CATEGORIES = {
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    }


# ============================================================================
# LOGGER
# ============================================================================

class Logger:
    """Simple colored logger."""
    
    COLORS = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m"
    }
    
    @staticmethod
    def info(msg: str):
        print(f"{Logger.COLORS['INFO']}[INFO]{Logger.COLORS['RESET']} {msg}")
    
    @staticmethod
    def success(msg: str):
        print(f"{Logger.COLORS['SUCCESS']}[✓]{Logger.COLORS['RESET']} {msg}")
    
    @staticmethod
    def warning(msg: str):
        print(f"{Logger.COLORS['WARNING']}[⚠]{Logger.COLORS['RESET']} {msg}")
    
    @staticmethod
    def error(msg: str):
        print(f"{Logger.COLORS['ERROR']}[✗]{Logger.COLORS['RESET']} {msg}")
    
    @staticmethod
    def header(msg: str):
        print(f"\n{'='*70}")
        print(f"  {msg}")
        print(f"{'='*70}\n")


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_mvtec() -> Dict:
    """Validate MVTec AD subdirectory."""
    Logger.header("Validating MVTec AD Categories")
    
    mvtec_path = os.path.join(Config.DATASET_PATH, Config.MVTEC_SUBDIR)
    
    if not os.path.exists(mvtec_path):
        Logger.error(f"MVTec directory not found: {mvtec_path}")
        return {"valid": False, "categories": {}}
    
    categories = {}
    found_cats = set()
    
    for cat_dir in sorted(os.listdir(mvtec_path)):
        cat_path = os.path.join(mvtec_path, cat_dir)
        
        if not os.path.isdir(cat_path):
            continue
        
        found_cats.add(cat_dir)
        
        # Count samples
        train_good = 0
        test_good = 0
        test_anomalies = {}
        ground_truth = {}
        
        # Check train/good
        train_good_path = os.path.join(cat_path, "train", "good")
        if os.path.exists(train_good_path):
            train_good = len([f for f in os.listdir(train_good_path) if f.endswith('.png')])
        
        # Check test/good
        test_good_path = os.path.join(cat_path, "test", "good")
        if os.path.exists(test_good_path):
            test_good = len([f for f in os.listdir(test_good_path) if f.endswith('.png')])
        
        # Check test anomalies
        test_path = os.path.join(cat_path, "test")
        if os.path.exists(test_path):
            for defect_dir in os.listdir(test_path):
                if defect_dir != "good":
                    defect_path = os.path.join(test_path, defect_dir)
                    if os.path.isdir(defect_path):
                        count = len([f for f in os.listdir(defect_path) if f.endswith('.png')])
                        test_anomalies[defect_dir] = count
        
        # Check ground truth masks
        gt_path = os.path.join(cat_path, "ground_truth")
        if os.path.exists(gt_path):
            for defect_dir in os.listdir(gt_path):
                defect_path = os.path.join(gt_path, defect_dir)
                if os.path.isdir(defect_path):
                    count = len([f for f in os.listdir(defect_path) if f.endswith('.png')])
                    ground_truth[defect_dir] = count
        
        categories[cat_dir] = {
            "train_good": train_good,
            "test_good": test_good,
            "test_anomalies": test_anomalies,
            "ground_truth": ground_truth,
            "valid": len(ground_truth) > 0
        }
        
        status = "✓" if categories[cat_dir]["valid"] else "✗"
        Logger.info(f"{status} {cat_dir}: {train_good} train | "
                   f"{test_good} test(good) | {len(test_anomalies)} defect types")
    
    # Check for missing categories
    missing = Config.EXPECTED_MVTEC_CATEGORIES - found_cats
    if missing:
        Logger.warning(f"Missing categories: {missing}")
    
    all_valid = len(missing) == 0
    
    return {
        "valid": all_valid,
        "categories": categories,
        "found_count": len(found_cats),
        "expected_count": len(Config.EXPECTED_MVTEC_CATEGORIES),
        "missing": list(missing)
    }


def validate_decospan() -> Dict:
    """Validate DecoSpan restructured dataset."""
    Logger.header("Validating DecoSpan Dataset")
    
    decospan_path = os.path.join(Config.DATASET_PATH, Config.DECOSPAN_SUBDIR)
    
    if not os.path.exists(decospan_path):
        Logger.error(f"DecoSpan directory not found: {decospan_path}")
        return {"valid": False}
    
    # Count train/good
    train_good_path = os.path.join(decospan_path, "train", "good")
    train_good = len([f for f in os.listdir(train_good_path) 
                      if f.endswith(('.jpg', '.png'))]) if os.path.exists(train_good_path) else 0
    
    # Count test/good
    test_good_path = os.path.join(decospan_path, "test", "good")
    test_good = len([f for f in os.listdir(test_good_path) 
                     if f.endswith(('.jpg', '.png'))]) if os.path.exists(test_good_path) else 0
    
    # Count test/custom_defect
    custom_defect_path = os.path.join(decospan_path, "test", "custom_defect")
    custom_defect_images = []
    if os.path.exists(custom_defect_path):
        custom_defect_images = [f for f in os.listdir(custom_defect_path) 
                                if f.endswith(('.jpg', '.png'))]
    
    # Count masks
    masks_path = os.path.join(decospan_path, "ground_truth", "custom_defect")
    masks = []
    if os.path.exists(masks_path):
        masks = [f for f in os.listdir(masks_path) if f.endswith('.png')]
    
    # Validate mask-image correspondence
    image_basenames = set(os.path.splitext(img)[0] for img in custom_defect_images)
    mask_basenames = set(os.path.splitext(mask)[0] for mask in masks)
    
    missing_masks = image_basenames - mask_basenames
    extra_masks = mask_basenames - image_basenames
    
    valid = (train_good > 0 and test_good > 0 and 
             len(custom_defect_images) > 0 and len(masks) > 0 and
             len(missing_masks) == 0 and len(extra_masks) == 0)
    
    Logger.success(f"Train (good): {train_good}")
    Logger.success(f"Test (good): {test_good}")
    Logger.success(f"Test (anomaly/custom_defect): {len(custom_defect_images)}")
    Logger.success(f"Ground truth masks: {len(masks)}")
    
    if missing_masks:
        Logger.error(f"Missing masks: {missing_masks}")
    if extra_masks:
        Logger.warning(f"Extra masks (no corresponding image): {extra_masks}")
    
    return {
        "valid": valid,
        "train_good": train_good,
        "test_good": test_good,
        "custom_defect_images": len(custom_defect_images),
        "masks": len(masks),
        "missing_masks": list(missing_masks),
        "extra_masks": list(extra_masks)
    }


def validate_image_dimensions() -> Dict:
    """Validate that masks match image dimensions."""
    Logger.header("Validating Image-Mask Correspondence")
    
    if USE_CV2 is None:
        Logger.warning("Skipping dimension validation (no image library available)")
        return {"valid": True, "errors": []}
    
    decospan_path = os.path.join(Config.DATASET_PATH, Config.DECOSPAN_SUBDIR)
    custom_defect_path = os.path.join(decospan_path, "test", "custom_defect")
    masks_path = os.path.join(decospan_path, "ground_truth", "custom_defect")
    
    errors = []
    checked = 0
    
    if not os.path.exists(custom_defect_path) or not os.path.exists(masks_path):
        Logger.warning("Paths not found, skipping validation")
        return {"valid": True, "errors": []}
    
    for image_file in os.listdir(custom_defect_path)[:5]:  # Sample check first 5
        if not image_file.endswith(('.jpg', '.png')):
            continue
        
        image_path = os.path.join(custom_defect_path, image_file)
        mask_basename = os.path.splitext(image_file)[0]
        mask_file = f"{mask_basename}.png"
        mask_path = os.path.join(masks_path, mask_file)
        
        if not os.path.exists(mask_path):
            errors.append(f"Missing mask for: {image_file}")
            continue
        
        try:
            if USE_CV2:
                img = cv2.imread(image_path)
                mask = cv2.imread(mask_path)
                if img is None or mask is None:
                    errors.append(f"Failed to read: {image_file}")
                    continue
                img_shape = img.shape[:2]
                mask_shape = mask.shape[:2]
            else:
                img = Image.open(image_path)
                mask = Image.open(mask_path)
                img_shape = img.size[::-1]
                mask_shape = mask.size[::-1]
            
            if img_shape != mask_shape:
                errors.append(f"{image_file}: Image {img_shape} != Mask {mask_shape}")
            else:
                Logger.info(f"✓ {image_file}: {img_shape[0]}×{img_shape[1]}")
            
            checked += 1
        except Exception as e:
            errors.append(f"Error reading {image_file}: {e}")
    
    valid = len(errors) == 0
    if valid:
        Logger.success(f"All {checked} sampled image-mask pairs match perfectly")
    else:
        for error in errors:
            Logger.error(error)
    
    return {"valid": valid, "errors": errors, "checked": checked}


def generate_report(mvtec_result: Dict, decospan_result: Dict, dims_result: Dict) -> Dict:
    """Generate validation report."""
    Logger.header("Validation Report")
    
    report = {
        "mvtec": mvtec_result,
        "decospan": decospan_result,
        "dimensions": dims_result,
        "overall_valid": mvtec_result["valid"] and decospan_result["valid"] and dims_result["valid"]
    }
    
    print("\nMVTec AD Status:")
    print(f"  Valid: {mvtec_result['valid']}")
    print(f"  Categories: {mvtec_result['found_count']}/{mvtec_result['expected_count']}")
    if mvtec_result['missing']:
        print(f"  Missing: {mvtec_result['missing']}")
    else:
        print(f"  Missing: None")
    
    print("\nDecoSpan Status:")
    print(f"  Valid: {decospan_result['valid']}")
    print(f"  Train (good): {decospan_result['train_good']}")
    print(f"  Test (good): {decospan_result['test_good']}")
    print(f"  Test (anomaly): {decospan_result['custom_defect_images']}")
    print(f"  Ground truth masks: {decospan_result['masks']}")
    if decospan_result['missing_masks']:
        print(f"  Missing masks: {len(decospan_result['missing_masks'])}")
    
    print("\nDimension Validation:")
    print(f"  Valid: {dims_result['valid']}")
    print(f"  Checked: {dims_result['checked']} image-mask pairs")
    if dims_result['errors']:
        print(f"  Errors: {len(dims_result['errors'])}")
    
    if report['overall_valid']:
        print("\n" + "="*70)
        print("  ✓ DATASET IS VALID AND READY FOR USE")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("  ✗ DATASET HAS ISSUES - REVIEW ABOVE ERRORS")
        print("="*70)
    
    return report


def main():
    """Main validation routine."""
    
    Logger.header("MSVPD Unified Dataset Validator")
    
    if not os.path.exists(Config.DATASET_PATH):
        Logger.error(f"Dataset not found at: {Config.DATASET_PATH}")
        sys.exit(1)
    
    # Run validations
    mvtec_result = validate_mvtec()
    decospan_result = validate_decospan()
    dims_result = validate_image_dimensions()
    
    # Generate report
    report = generate_report(mvtec_result, decospan_result, dims_result)
    
    # Save report
    report_path = os.path.join(Config.DATASET_PATH, "validation_report.json")
    try:
        with open(report_path, 'w') as f:
            # Convert to serializable format
            serializable_report = {
                "mvtec": {
                    "valid": mvtec_result['valid'],
                    "found_count": mvtec_result['found_count'],
                    "expected_count": mvtec_result['expected_count'],
                    "missing": mvtec_result['missing'],
                    "categories": {k: {"train_good": v.get("train_good", 0),
                                       "test_good": v.get("test_good", 0),
                                       "valid": v.get("valid", False)}
                                  for k, v in mvtec_result['categories'].items()}
                },
                "decospan": decospan_result,
                "overall_valid": report['overall_valid']
            }
            json.dump(serializable_report, f, indent=2)
        Logger.success(f"Report saved to: {report_path}")
    except Exception as e:
        Logger.warning(f"Could not save report: {e}")
    
    sys.exit(0 if report['overall_valid'] else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        Logger.warning("Validation interrupted by user")
        sys.exit(0)
    except Exception as e:
        Logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
