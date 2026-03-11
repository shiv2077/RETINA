#!/usr/bin/env python3
"""
MSVPD Dataset Merger - Pre-Flight Check
========================================

Verifies all prerequisites before running the full merge operation.
Checks for:
  - Python version compatibility
  - Required directories exist
  - Disk space availability
  - Image library availability
  - File permissions

Author: Data Engineering Pipeline
Date: 2026
"""

import os
import sys
import shutil
from pathlib import Path


class Colors:
    """Terminal colors."""
    RESET = '\033[0m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'


def print_header(msg):
    """Print section header."""
    print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.CYAN}  {msg}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")


def check_pass(msg):
    """Print success message."""
    print(f"{Colors.GREEN}[✓]{Colors.RESET} {msg}")


def check_fail(msg):
    """Print failure message."""
    print(f"{Colors.RED}[✗]{Colors.RESET} {msg}")


def check_warn(msg):
    """Print warning message."""
    print(f"{Colors.YELLOW}[⚠]{Colors.RESET} {msg}")


def check_info(msg):
    """Print info message."""
    print(f"{Colors.BLUE}[i]{Colors.RESET} {msg}")


def main():
    """Run pre-flight checks."""
    
    print_header("MSVPD Dataset Merger - Pre-Flight Check")
    
    all_pass = True
    
    # ========================================================================
    # CHECK 1: Python Version
    # ========================================================================
    print_header("CHECK 1: Python Version")
    
    py_version = sys.version_info
    
    if py_version.major == 3 and py_version.minor >= 8:
        check_pass(f"Python 3.{py_version.minor}.{py_version.micro} (>= 3.8 required)")
    else:
        check_fail(f"Python 3.{py_version.minor}.{py_version.micro} (>= 3.8 required)")
        all_pass = False
    
    # ========================================================================
    # CHECK 2: Required Modules
    # ========================================================================
    print_header("CHECK 2: Required Python Modules")
    
    required_modules = ['shutil', 'os', 'pathlib', 'json', 'time']
    for module in required_modules:
        try:
            __import__(module)
            check_pass(f"{module} (built-in, available)")
        except ImportError:
            check_fail(f"{module} (missing - should never happen)")
            all_pass = False
    
    # ========================================================================
    # CHECK 3: Image Processing Library
    # ========================================================================
    print_header("CHECK 3: Image Processing Library")
    
    has_cv2 = False
    has_pil = False
    
    try:
        import cv2
        check_pass(f"OpenCV (cv2) found - version {cv2.__version__}")
        has_cv2 = True
    except ImportError:
        check_warn("OpenCV (cv2) not found")
    
    try:
        from PIL import Image
        import PIL
        check_pass(f"Pillow (PIL) found - version {PIL.__version__}")
        has_pil = True
    except ImportError:
        check_warn("Pillow (PIL) not found")
    
    if not (has_cv2 or has_pil):
        check_fail("Neither OpenCV nor Pillow available (at least one required)")
        check_info("Install with: pip install opencv-python  OR  pip install Pillow")
        all_pass = False
    else:
        check_pass("Image processing library available")
    
    # ========================================================================
    # CHECK 4: Source Datasets
    # ========================================================================
    print_header("CHECK 4: Source Datasets")
    
    mvtec_path = Path("./mvtec_anomaly_detection")
    decospan_path = Path("./decospan_small")
    
    mvtec_exists = mvtec_path.exists()
    decospan_exists = decospan_path.exists()
    
    if mvtec_exists:
        check_pass(f"MVTec AD found at: {mvtec_path.resolve()}")
        
        # Count categories
        mvtec_categories = [d for d in mvtec_path.iterdir() 
                           if d.is_dir() and not d.name.startswith('.')]
        check_info(f"Found {len(mvtec_categories)} categories")
        
        if len(mvtec_categories) != 15:
            check_warn(f"Expected 15 categories, found {len(mvtec_categories)}")
    else:
        check_fail(f"MVTec AD not found at: {mvtec_path.resolve()}")
        check_info("From RETINA project root: mvtec_anomaly_detection/")
        all_pass = False
    
    if decospan_exists:
        check_pass(f"DecoSpan found at: {decospan_path.resolve()}")
        
        # Check structure
        train_path = decospan_path / "train"
        test_path = decospan_path / "test"
        
        if train_path.exists():
            check_info("train/ subdirectory found")
        else:
            check_warn("train/ subdirectory not found")
        
        if test_path.exists():
            check_info("test/ subdirectory found")
        else:
            check_warn("test/ subdirectory not found")
    else:
        check_fail(f"DecoSpan not found at: {decospan_path.resolve()}")
        check_info("From RETINA project root: decospan_small/")
        all_pass = False
    
    # ========================================================================
    # CHECK 5: File Permissions
    # ========================================================================
    print_header("CHECK 5: File Permissions")
    
    readable_mvtec = os.access(mvtec_path, os.R_OK) if mvtec_exists else None
    readable_decospan = os.access(decospan_path, os.R_OK) if decospan_exists else None
    
    if readable_mvtec:
        check_pass("MVTec AD readable")
    elif mvtec_exists:
        check_fail("MVTec AD not readable - check permissions")
        all_pass = False
    
    if readable_decospan:
        check_pass("DecoSpan readable")
    elif decospan_exists:
        check_fail("DecoSpan not readable - check permissions")
        all_pass = False
    
    # ========================================================================
    # CHECK 6: Disk Space
    # ========================================================================
    print_header("CHECK 6: Disk Space")
    
    try:
        stat = shutil.disk_usage(".")
        total_gb = stat.total / (1024**3)
        free_gb = stat.free / (1024**3)
        used_gb = stat.used / (1024**3)
        
        check_info(f"Total: {total_gb:.1f} GB")
        check_info(f"Used:  {used_gb:.1f} GB")
        check_info(f"Free:  {free_gb:.1f} GB")
        
        # Estimate required space
        required_gb = 7  # ~6GB MVTec + 1GB margin
        
        if free_gb > required_gb:
            check_pass(f"Sufficient disk space ({free_gb:.1f} GB free, {required_gb} GB required)")
        else:
            check_fail(f"Low disk space ({free_gb:.1f} GB free, {required_gb} GB required)")
            all_pass = False
    except Exception as e:
        check_warn(f"Could not check disk space: {e}")
    
    # ========================================================================
    # CHECK 7: Target Directory
    # ========================================================================
    print_header("CHECK 7: Target Directory")
    
    target_path = Path("./MSVPD_Unified_Dataset")
    
    if target_path.exists():
        check_warn(f"Target directory already exists: {target_path.resolve()}")
        check_info("You will be asked to overwrite when running merge_datasets.py")
    else:
        check_pass(f"Target path is available: {target_path.resolve()}")
    
    # ========================================================================
    # CHECK 8: Current Working Directory
    # ========================================================================
    print_header("CHECK 8: Current Working Directory")
    
    cwd = Path.cwd()
    retina_marker = cwd / "README.md"  # Assuming RETINA has README.md
    
    check_info(f"Current directory: {cwd}")
    
    if retina_marker.exists():
        check_pass("Appears to be RETINA project root")
    else:
        check_warn("README.md not found - might not be RETINA project root")
        check_info("Make sure to run from /path/to/RETINA directory")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_header("Summary")
    
    if all_pass:
        print(f"{Colors.GREEN}✓ All checks passed!{Colors.RESET}")
        print("\nYou can now run:")
        print(f"  {Colors.CYAN}python merge_datasets.py{Colors.RESET}")
        print("\nThen validate with:")
        print(f"  {Colors.CYAN}python validate_dataset.py{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}✗ Some checks failed{Colors.RESET}")
        print("\nPlease fix the issues above before running merge_datasets.py")
        print("\nCommon fixes:")
        print(f"  1. Install missing library: {Colors.CYAN}pip install opencv-python{Colors.RESET}")
        print(f"  2. Check dataset paths - run from RETINA root: {Colors.CYAN}cd /path/to/RETINA{Colors.RESET}")
        print(f"  3. Fix permissions: {Colors.CYAN}chmod -R 755 mvtec_anomaly_detection/ decospan_small/{Colors.RESET}")
        print(f"  4. Free up disk space if needed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
