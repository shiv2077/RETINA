#!/usr/bin/env python3
"""
MSVPD UNIFIED DATASET - QUICKSTART GUIDE
=========================================

This document explains how to merge MVTec AD and DecoSpan datasets
into a unified directory for use with anomalib and the RETINA pipeline.

Author: Data Engineering Team
Date: 2026
"""

# ============================================================================
# OVERVIEW
# ============================================================================

"""
What This Does
==============

This toolkit creates a unified directory structure from two datasets:

SOURCE DATASETS:
  1. mvtec_anomaly_detection/   (15 industrial product categories - ~6000 images)
  2. decospan_small/             (custom manufacturing dataset - 77 images)

TARGET DATASET:
  MSVPD_Unified_Dataset/
  ├── MVTec/                     (All 15 categories with original structure)
  │   ├── bottle/
  │   ├── cable/
  │   ├── ... (13 more)
  │   └── zipper/
  │
  └── Decospan/
      ├── train/
      │   └── good/              (20 images)
      ├── test/
      │   ├── good/              (31 images)
      │   └── custom_defect/     (26 anomaly images)
      └── ground_truth/
          └── custom_defect/     (26 PNG masks)


KEY FEATURES
============

✓ Non-destructive: Original datasets remain untouched
✓ Automatic restructuring: DecoSpan reorg happens seamlessly
✓ Mask generation: White PNG masks created for all anomalies
✓ Progress tracking: Detailed console output for all operations
✓ Validation: Built-in verification script to ensure integrity
✓ MLOps ready: Structured for both Stage 1 (VLM) and Stage 3 (BGAD)


DATA FLOW
=========

Original DecoSpan:                 Restructured DecoSpan:
├── train/                         ├── train/
│   ├── normal/ (20)    ──────→    │   └── good/ (20)
│   └── anomaly/ (14)      ↓
│                          └──→    ├── test/
└── test/                          │   ├── good/ (31)
    ├── normal/ (31)     ──────→   │   └── custom_defect/ (12 + 14)
    └── anomaly/ (12)       ↓
                            └──→   └── ground_truth/
                                       └── custom_defect/ (26 masks)

Total images in unified dataset:
  MVTec: ~6000+ images across 15 categories
  DecoSpan: 77 images (78 with generated masks)
"""

# ============================================================================
# QUICK START
# ============================================================================

"""
1. MERGE THE DATASETS
=====================

$ python merge_datasets.py

What happens:
  - Creates MSVPD_Unified_Dataset/ directory
  - Copies all 15 MVTec categories
  - Restructures DecoSpan files
  - Generates 26 white PNG masks
  - Prints detailed progress for each step
  - Estimated time: 2-5 minutes (depending on disk speed)

Expected output:
  [INFO] MVTec AD found at: ./mvtec_anomaly_detection
  [INFO] DecoSpan Small found at: ./decospan_small
  [✓ OK] Created directory: ./MSVPD_Unified_Dataset/MVTec/
  [✓ OK] Copied {category}
  ... (15 categories)
  [✓ OK] Copied 20 files: train/normal -> train/good
  [✓ OK] Copied 31 files: test/normal -> test/good
  [✓ OK] Copied 14 files: train/anomaly -> test/custom_defect
  [✓ OK] Copied 12 files: test/anomaly -> test/custom_defect
  [INFO] Mask N: {filename}.png (1024×1024)
  ... (26 masks)
  [✓ COMPLETE] - Elapsed Time: 3.45 seconds


2. VALIDATE THE DATASET
=======================

$ python validate_dataset.py

What happens:
  - Checks MVTec structure and file counts
  - Checks DecoSpan restructuring
  - Verifies image-mask correspondence
  - Generates validation_report.json
  - Reports any issues

Expected output:
  [INFO] bottle: 209 train | 20 test(good) | 3 defect types
  [INFO] cable: 320 train | 20 test(good) | 2 defect types
  ... (13 more categories)
  
  [✓] Train (good): 20
  [✓] Test (good): 31
  [✓] Test (anomaly/custom_defect): 26
  [✓] Ground truth masks: 26
  
  [✓] All 26 sampled image-mask pairs match perfectly
  
  ✓ DATASET IS VALID AND READY FOR USE


3. USE IN YOUR PIPELINE
=======================

# Option A: Stage 1 (VLM Benchmarking) - Use MVTec
from anomalib.data import MVTecDataModule

dm = MVTecDataModule(
    root="./MSVPD_Unified_Dataset/MVTec",
    category="bottle",
    image_size=256,
    batch_size=32,
)

# Option B: Stage 3 (BGAD Supervised) - Use DecoSpan
from anomalib.data import Folder

dm = Folder(
    root="./MSVPD_Unified_Dataset/Decospan",
    image_size=256,
    train_batch_size=16,
    eval_batch_size=32,
    normal_dir="train/good",
    abnormal_dir="test/custom_defect",
    mask_dir="ground_truth/custom_defect",
)

# Option C: Hybrid Pipeline - Sequential Stages
from RETINA.pipeline import PipelineService

pipeline = PipelineService()

# Stage 1 on MVTec
for category in ["bottle", "cable", "capsule"]:
    pipeline.run_stage1_unsupervised(
        category=category,
        data_root="./MSVPD_Unified_Dataset/MVTec"
    )

# Stage 3 on DecoSpan
pipeline.run_stage3_supervised(
    data_root="./MSVPD_Unified_Dataset/Decospan",
    epochs=30
)
"""

# ============================================================================
# DETAILED STEPS
# ============================================================================

"""
STEP 1: REQUIREMENTS
====================

Python 3.8+
numpy
shutil (built-in)
os (built-in)
pathlib (built-in)

Optional (for image operations):
opencv-python (cv2)  OR  Pillow (PIL)

Install:
  $ pip install opencv-python
  # or
  $ pip install Pillow


STEP 2: VERIFY DATASETS EXIST
=============================

Before running, make sure both source datasets exist:

$ ls -la mvtec_anomaly_detection/
  Should show: bottle, cable, capsule, ... zipper, license.txt, readme.txt

$ ls -la decospan_small/
  Should show: train/, test/


STEP 3: RUN MERGER SCRIPT
=========================

$ python merge_datasets.py

This will:
  [VALIDATION]
    ✓ Check mvtec_anomaly_detection exists
    ✓ Check decospan_small exists
    ✓ Count 15 categories
  
  [DIRECTORY CREATION]
    ✓ Create MSVPD_Unified_Dataset/
    ✓ Create MSVPD_Unified_Dataset/MVTec/
    ✓ Create MSVPD_Unified_Dataset/Decospan/train/good/
    ✓ Create MSVPD_Unified_Dataset/Decospan/test/good/
    ✓ Create MSVPD_Unified_Dataset/Decospan/test/custom_defect/
    ✓ Create MSVPD_Unified_Dataset/Decospan/ground_truth/custom_defect/
  
  [MVTEC TRANSFER]
    ✓ Transfer bottle (209 train → 83 test)
    ✓ Transfer cable (320 train → 80 test)
    ... (13 more)
    ✓ Successfully transferred 15/15 categories
  
  [DECOSPAN RESTRUCTURE]
    ✓ Copy train/normal → train/good (20 files)
    ✓ Copy test/normal → test/good (31 files)
    ✓ Copy train/anomaly → test/custom_defect (14 files)
    ✓ Copy test/anomaly → test/custom_defect (12 files)
    Total anomaly images: 26
  
  [MASK GENERATION]
    ✓ Mask 1: PO23-03227_4_1_img001_patch011.png (256×256)
    ✓ Mask 2: PO23-03227_4_1_img001_patch012.png (256×256)
    ... (24 more)
    ✓ Generated 26 masks (0 errors)


STEP 4: VALIDATE DATASET
========================

$ python validate_dataset.py

This will:
  [MVTec Validation]
    ✓ bottle: 209 train | 20 test(good) | 3 defect types
    ✓ cable: 320 train | 20 test(good) | 2 defect types
    ... (13 categories)
  
  Found 15/15 expected categories
  
  [DecoSpan Validation]
    ✓ Train (good): 20
    ✓ Test (good): 31
    ✓ Test (anomaly/custom_defect): 26
    ✓ Ground truth masks: 26
  
  [Dimension Validation]
    ✓ PO23-03227_4_1_img001_patch011.jpg: 256×256
    ✓ PO23-03227_4_1_img001_patch012.jpg: 256×256
    ... (checked 5 samples)
    ✓ All 5 sampled image-mask pairs match perfectly
  
  ✓ DATASET IS VALID AND READY FOR USE
  
  Report saved to: MSVPD_Unified_Dataset/validation_report.json


STEP 5: USE IN YOUR CODE
========================

# For MVTec only
from pathlib import Path
mvtec_root = Path("./MSVPD_Unified_Dataset/MVTec")
for category in mvtec_root.iterdir():
    print(f"Training on {category.name}")

# For DecoSpan only
decospan_root = Path("./MSVPD_Unified_Dataset/Decospan")
train_good = list((decospan_root / "train" / "good").glob("*.jpg"))
test_good = list((decospan_root / "test" / "good").glob("*.jpg"))
anomalies = list((decospan_root / "test" / "custom_defect").glob("*.jpg"))
masks = list((decospan_root / "ground_truth" / "custom_defect").glob("*.png"))

print(f"Train normal: {len(train_good)}")
print(f"Test normal: {len(test_good)}")
print(f"Anomalies: {len(anomalies)}")
print(f"Masks: {len(masks)}")
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
ISSUE: "mvtec_anomaly_detection not found"
SOLUTION: 
  Make sure you're in the correct directory (RETINA project root)
  $ pwd
  # Should show: /path/to/RETINA
  $ ls mvtec_anomaly_detection/
  # Should list: bottle, cable, ...

ISSUE: "Permission denied when copying files"
SOLUTION:
  Check folder permissions:
  $ chmod -R 755 mvtec_anomaly_detection/
  $ chmod -R 755 decospan_small/

ISSUE: "Neither cv2 nor PIL available"
SOLUTION:
  Install image library:
  $ pip install opencv-python
  # The script will fall back to PIL if cv2 is not available

ISSUE: "Missing masks in validation"
SOLUTION:
  This means image-mask correspondence is broken
  1. Re-run merge_datasets.py
  2. Check if MSVPD_Unified_Dataset already exists
  3. If yes, you'll be asked to overwrite - answer 'y'

ISSUE: "Dimension mismatch between image and mask"
SOLUTION:
  This is usually caused by corruption during copy
  1. Delete MSVPD_Unified_Dataset/Decospan/ground_truth/
  2. Re-run merge_datasets.py
  3. Answer 'y' to overwrite when prompted

ISSUE: Script hangs during file copy
SOLUTION:
  This can happen with very large datasets
  1. Check disk space: $ df -h
  2. Check if disk is full or slow
  3. Close other I/O intensive programs
  4. Re-run script - it will skip existing files

ISSUE: "validate_dataset.py shows errors"
SOLUTION:
  Check the generated validation_report.json:
  $ cat MSVPD_Unified_Dataset/validation_report.json
  
  Look for:
    "missing": [] (should be empty)
    "valid": true (should be true for all sections)
"""

# ============================================================================
# ADVANCED USAGE
# ============================================================================

"""
CUSTOM CONFIGURATION
====================

Edit merge_datasets.py Config class:

class Config:
    MVTEC_SOURCE = "./mvtec_anomaly_detection"          # Change source path
    DECOSPAN_SOURCE = "./decospan_small"                # Change source path
    UNIFIED_TARGET = "./MSVPD_Unified_Dataset"          # Change target path
    MVTEC_SUBDIR = "MVTec"                              # Change subdirectory name
    DECOSPAN_SUBDIR = "Decospan"                        # Change subdirectory name
    MASK_VALUE = 255                                    # Change mask color (0-255)
    MASK_FORMAT = "png"                                 # Change output format


BATCH PROCESSING MULTIPLE DATASETS
===================================

# If you have multiple decospan variant folders:

for dataset_name in ["decospan_small", "decospan_medium", "decospan_large"]:
    config.DECOSPAN_SOURCE = f"./{dataset_name}"
    config.UNIFIED_TARGET = f"./MSVPD_{dataset_name}"
    main()  # Run merger


SYMLINK INSTEAD OF COPY (UNIX ONLY)
====================================

Edit merge_datasets.py to use symlinks (saves disk space):

# Change this line:
shutil.copytree(src_cat, dst_cat, dirs_exist_ok=False)

# To:
os.symlink(src_cat, dst_cat)

This creates symbolic links instead of copying files.
Requires UNIX-like system (Linux, macOS). Doesn't work on Windows.


MEMORY-EFFICIENT MODE
===================

For systems with limited RAM, process in batches:

# Edit merge_datasets.py Phase 3:
for category in sorted(mvtec_categories):
    shutil.copytree(...)
    # Add explicit garbage collection after each copy
    import gc
    gc.collect()
"""

# ============================================================================
# ARCHITECTURE REFERENCE
# ============================================================================

"""
RESULTING DIRECTORY TREE
========================

MSVPD_Unified_Dataset/
├── validation_report.json          # Generated by validate_dataset.py
│
├── MVTec/
│   ├── bottle/
│   │   ├── train/
│   │   │   └── good/ (209 images)
│   │   ├── test/
│   │   │   ├── good/ (20 images)
│   │   │   ├── broken_large/ (20)
│   │   │   ├── broken_small/ (22)
│   │   │   └── contamination/ (21)
│   │   └── ground_truth/
│   │       ├── broken_large/ (20 masks)
│   │       ├── broken_small/ (22 masks)
│   │       └── contamination/ (21 masks)
│   │
│   ├── cable/
│   ├── capsule/
│   ├── carpet/
│   ├── grid/
│   ├── hazelnut/
│   ├── leather/
│   ├── metal_nut/
│   ├── pill/
│   ├── screw/
│   ├── tile/
│   ├── toothbrush/
│   ├── transistor/
│   ├── wood/
│   └── zipper/
│
└── Decospan/
    ├── train/
    │   └── good/ (20 .jpg files)
    │       ├── PO22-33844_2_2_img000_patch009.jpg
    │       └── ... (19 more)
    │
    ├── test/
    │   ├── good/ (31 .jpg files)
    │   │   └── ... (31 images)
    │   │
    │   └── custom_defect/ (26 .jpg files)
    │       ├── PO23-08735_30_1_img005_patch032.jpg (train/anomaly)
    │       ├── PO23-09456_6_1_img006_patch219.jpg (train/anomaly)
    │       ├── ... (14 from train/anomaly)
    │       ├── PO23-03227_4_1_img001_patch011.jpg (test/anomaly)
    │       └── ... (12 from test/anomaly)
    │
    └── ground_truth/
        └── custom_defect/ (26 .png files)
            ├── PO23-08735_30_1_img005_patch032.png (mask)
            ├── PO23-09456_6_1_img006_patch219.png (mask)
            └── ... (24 more masks)


STATISTICS
==========

MVTec AD:
  Total categories: 15
  Total training samples (normal only): ~3000
  Total test samples: ~3000
  Average per category:
    - Training: 200 images
    - Test: 200 images (60% normal + 40% defects)
    - Defect types: 3-5 per category
  Total size: ~6GB

DecoSpan:
  Training: 20 normal + 14 anomaly = 34 images
  Test: 31 normal + 12 anomaly = 43 images
  Total: 77 images (+ 26 generated masks = 103 files)
  Total size: ~10MB

Unified Dataset:
  Total: ~6GB + 10MB ≈ 6.01GB
  File count: ~9000+
  Directories: 30+


COMPATIBILITY
=============

✓ anomalib >= 0.5.0
✓ PyTorch >= 1.10.0
✓ torchvision >= 0.11.0
✓ Python 3.8+
✓ Linux, macOS, Windows
✓ Works with both GPU and CPU
"""

# ============================================================================
# NEXT STEPS FOR RETINA PIPELINE
# ============================================================================

"""
AFTER MERGING AND VALIDATING
=============================

1. Update pipeline configuration:
   
   # src/backend/config.py
   UNIFIED_DATASET_PATH = "./MSVPD_Unified_Dataset"
   
   # For MVTec:
   MVTEC_PATH = f"{UNIFIED_DATASET_PATH}/MVTec"
   
   # For DecoSpan:
   DECOSPAN_PATH = f"{UNIFIED_DATASET_PATH}/Decospan"


2. Update Stage 1 (VLM Benchmarking):
   
   # Iterate over all MVTec categories
   for category in ["bottle", "cable", ..., "zipper"]:
       pipeline.run_stage1_unsupervised(
           category=category,
           data_root=MVTEC_PATH
       )


3. Update Stage 3 (BGAD Supervised):
   
   # Use DecoSpan as labeled training data
   pipeline.run_stage3_supervised(
       train_dir=f"{DECOSPAN_PATH}/train/good",
       test_dir=f"{DECOSPAN_PATH}/test",
       mask_dir=f"{DECOSPAN_PATH}/ground_truth/custom_defect",
       epochs=30
   )


4. Run cross-domain evaluation:
   
   # Train on MVTec, test on DecoSpan (out-of-distribution test)
   for category in ["bottle", ...]:
       patchcore = train_on_mvtec(category, MVTEC_PATH)
       
       # Evaluate on decospan
       auroc = patchcore.evaluate_on_decospan(DECOSPAN_PATH)


5. Commit to git:
   
   $ git add MSVPD_Unified_Dataset/
   $ git add merge_datasets.py validate_dataset.py
   $ git commit -m "Add unified dataset structure and tooling"
"""

print(__doc__)
