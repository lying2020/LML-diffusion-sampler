#!/usr/bin/env python3
"""
Prepare COCO Reference Dataset for CMMD Evaluation

This script helps prepare reference images from MS COCO dataset for CMMD evaluation.
It can:
1. Download COCO validation set
2. Extract images matching specific prompts
3. Organize reference images for CMMD evaluation

Usage:
    # Download COCO validation set
    python evaluations/prepare_coco_reference.py --download --output_dir data/coco_reference

    # Extract images matching prompts (if you have COCO annotations)
    python evaluations/prepare_coco_reference.py \
        --prompts_file evaluations/coco_prompts/coco_top_40_prompts_backup.json \
        --coco_images_dir data/coco/val2017 \
        --coco_annotations data/coco/annotations/instances_val2017.json \
        --output_dir data/coco_reference_matched
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import urllib.request
import zipfile
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())


def download_coco_validation(output_dir: str):
    """
    Download MS COCO 2017 validation set.

    Args:
        output_dir: Directory to save downloaded images
    """
    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = os.path.join(output_dir, "val2017.zip")
    extract_dir = output_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"📥 Downloading COCO 2017 validation set...")
    print(f"   URL: {url}")
    print(f"   Output: {output_dir}")

    try:
        # Download with progress bar
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\r   Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end='')

        urllib.request.urlretrieve(url, zip_path, show_progress)
        print("\n✅ Download complete!")

        # Extract
        print(f"📦 Extracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Remove zip file
        os.remove(zip_path)
        print("✅ Extraction complete!")
        print(f"📁 Images are in: {os.path.join(extract_dir, 'val2017')}")

    except Exception as e:
        print(f"❌ Error downloading COCO dataset: {e}")
        print("   You can manually download from: http://images.cocodataset.org/zips/val2017.zip")
        raise


def load_coco_annotations(annotations_file: str) -> Dict:
    """Load COCO annotations from JSON file."""
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    return annotations


def find_matching_images(
    prompts_file: str,
    coco_images_dir: str,
    coco_annotations_file: Optional[str] = None,
    output_dir: str = None
) -> List[str]:
    """
    Find COCO images that match the prompts.

    This is a simplified version. For exact matching, you would need:
    1. COCO annotations with captions
    2. Text similarity matching between prompts and captions

    Args:
        prompts_file: Path to prompts JSON file
        coco_images_dir: Directory containing COCO images
        coco_annotations_file: Optional COCO annotations file
        output_dir: Optional directory to copy matching images

    Returns:
        List of matching image paths
    """
    # Load prompts
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    print(f"📝 Loaded {len(prompts)} prompts")

    if coco_annotations_file and os.path.exists(coco_annotations_file):
        # Try to match using annotations
        print("🔍 Attempting to match images using COCO annotations...")
        annotations = load_coco_annotations(coco_annotations_file)

        # This is a placeholder - actual matching would require:
        # 1. Loading captions from annotations
        # 2. Computing text similarity between prompts and captions
        # 3. Selecting best matches

        print("⚠️  Exact matching not implemented. Using all COCO images as reference.")
        print("   For CMMD, using all COCO validation images is acceptable.")

    # Collect all COCO images
    coco_path = Path(coco_images_dir)
    image_extensions = ['jpg', 'jpeg', 'png']
    all_images = []

    for ext in image_extensions:
        all_images.extend(coco_path.glob(f'*.{ext}'))
        all_images.extend(coco_path.glob(f'*.{ext.upper()}'))

    all_images = sorted([str(img) for img in all_images])
    print(f"✅ Found {len(all_images)} COCO images")

    # Copy to output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📋 Copying images to {output_dir}...")

        for img_path in tqdm(all_images, desc="Copying images"):
            img_name = os.path.basename(img_path)
            dst_path = os.path.join(output_dir, img_name)
            shutil.copy2(img_path, dst_path)

        print(f"✅ Copied {len(all_images)} images to {output_dir}")

    return all_images


def main():
    parser = argparse.ArgumentParser(
        description="Prepare COCO reference dataset for CMMD evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='Download COCO 2017 validation set'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/liying/Documents/dataset/coco/val2014',
        help='Output directory for reference images (default: /home/liying/Documents/dataset/coco/val2014)'
    )

    parser.add_argument(
        '--prompts_file',
        type=str,
        default=None,
        help='Path to prompts JSON file (for matching images)'
    )

    parser.add_argument(
        '--coco_images_dir',
        type=str,
        default='/home/liying/Documents/dataset/coco/val2014',
        help='Directory containing COCO images (default: /home/liying/Documents/dataset/coco/val2014)'
    )

    parser.add_argument(
        '--coco_annotations',
        type=str,
        default=None,
        help='Path to COCO annotations JSON file (for matching)'
    )

    args = parser.parse_args()

    if args.download:
        # Download COCO validation set
        download_coco_validation(args.output_dir)

    elif args.coco_images_dir:
        # Extract matching images
        if not args.prompts_file:
            print("⚠️  --prompts_file is recommended for matching images")
            print("   Will use all COCO images as reference")

        find_matching_images(
            prompts_file=args.prompts_file or 'evaluations/coco_prompts/coco_top_40_prompts_backup.json',
            coco_images_dir=args.coco_images_dir,
            coco_annotations_file=args.coco_annotations,
            output_dir=args.output_dir
        )

    else:
        parser.print_help()
        print("\n💡 Usage examples:")
        print("   # Download COCO validation set:")
        print("   python evaluations/prepare_coco_reference.py --download --output_dir data/coco_reference")
        print("\n   # Use existing COCO images:")
        print("   python evaluations/prepare_coco_reference.py \\")
        print("       --coco_images_dir data/coco/val2017 \\")
        print("       --output_dir data/coco_reference")


if __name__ == '__main__':
    main()
