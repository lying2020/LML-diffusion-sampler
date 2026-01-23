#!/usr/bin/env python3
"""
CMMD and CLIP-Score Evaluation Script

This script evaluates generated images using:
1. CMMD (Conditional Maximum Mean Discrepancy) - measures distribution distance
2. CLIP-Score - measures image-text alignment

Usage:
    python evaluations/cmmd_clipscore_eval.py \
        --output_dir output/coco_stable-diffusion-2-base/steps_20/dpm_lm \
        --prompts_file evaluations/coco_prompts/coco_top_40_prompts_backup.json
"""

import os
import sys
import json
import argparse
import glob
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: clip library not found. Please install: pip install git+https://github.com/openai/CLIP.git")

try:
    from transformers import CLIPModel, CLIPProcessor
    HF_CLIP_AVAILABLE = True
except ImportError:
    HF_CLIP_AVAILABLE = False
    print("Warning: transformers not found. Will use standard CLIP loading for local paths.")

try:
    from cmmd_pytorch import compute_cmmd
    CMMD_AVAILABLE = True
except ImportError:
    CMMD_AVAILABLE = False
    # Don't print warning here, will be handled when CMMD is actually needed

# Image extensions
IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp'}


def parse_image_filename(filename: str) -> Optional[Dict[str, any]]:
    """
    Parse image filename to extract metadata.

    Expected format: {index:05d}_{prompt_id}_{guidance}_{inference_steps}_{seed}_{sampler}.jpg
    Example: 00000_1_guidance7.5_inference10_seed6_dpm_lm.jpg

    Returns:
        Dictionary with parsed metadata or None if parsing fails
    """
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)

    # Pattern: index_prompt_id_guidanceX.X_inferenceX_seedX_sampler
    pattern = r'(\d+)_(\d+)_guidance([\d.]+)_inference(\d+)_seed(\d+)_(.+)'
    match = re.match(pattern, name)

    if match:
        return {
            'index': int(match.group(1)),
            'prompt_id': match.group(2),
            'guidance': float(match.group(3)),
            'inference_steps': int(match.group(4)),
            'seed': int(match.group(5)),
            'sampler': match.group(6),
            'filename': filename
        }

    # Try simpler pattern: index_prompt_id_...
    pattern2 = r'(\d+)_(\d+)_(.+)'
    match2 = re.match(pattern2, name)
    if match2:
        return {
            'index': int(match2.group(1)),
            'prompt_id': match2.group(2),
            'filename': filename
        }

    return None


def load_prompts(prompts_file: str) -> Dict[str, str]:
    """Load prompts from JSON file."""
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    return prompts


def collect_images(output_dir: str) -> List[str]:
    """Collect all image files from output directory."""
    image_files = []
    output_path = Path(output_dir)

    for ext in IMAGE_EXTENSIONS:
        image_files.extend(output_path.glob(f'*.{ext}'))
        image_files.extend(output_path.glob(f'*.{ext.upper()}'))

    # Sort by filename
    image_files = sorted([str(f) for f in image_files])
    return image_files


def load_clip_model(model_name_or_path: str = "ViT-L/14", device: str = "cuda"):
    """
    Load CLIP model from either:
    1. Local HuggingFace format path (if path exists)
    2. Standard CLIP model name (e.g., "ViT-L/14")

    Returns:
        (model, preprocess, processor, is_hf) tuple
        - is_hf: True if HuggingFace format, False if standard CLIP
    """
    # Check if it's a local path
    if os.path.exists(model_name_or_path):
        if HF_CLIP_AVAILABLE:
            try:
                print(f"🔄 Loading CLIP model from local path: {model_name_or_path}")
                model = CLIPModel.from_pretrained(model_name_or_path)
                processor = CLIPProcessor.from_pretrained(model_name_or_path)
                model = model.to(device)
                model.eval()

                # Create compatible preprocess function
                def preprocess_fn(image):
                    return processor(images=image, return_tensors="pt")['pixel_values'].to(device)

                return model, preprocess_fn, processor, True
            except Exception as e:
                print(f"⚠️  Failed to load from HuggingFace format: {e}")
                print("   Falling back to standard CLIP loading...")
        else:
            print("⚠️  transformers not available, cannot load HuggingFace format")
            print("   Falling back to standard CLIP loading...")

    # Use standard CLIP loading
    if not CLIP_AVAILABLE:
        raise ImportError("CLIP library not available. Please install: pip install git+https://github.com/openai/CLIP.git")

    print(f"🔄 Loading CLIP model: {model_name_or_path}")
    model, preprocess = clip.load(model_name_or_path, device=device)
    model.eval()
    return model, preprocess, None, False


def extract_image_features(
    image_paths: List[str],
    model,
    preprocess,
    device: str = "cuda",
    batch_size: int = 32,
    is_hf: bool = False
) -> torch.Tensor:
    """
    Extract CLIP image features from a list of image paths.

    Args:
        image_paths: List of image file paths
        model: CLIP model
        preprocess: Preprocessing function
        device: Device to run computation
        batch_size: Batch size
        is_hf: Whether using HuggingFace format model

    Returns:
        Tensor of shape (N, D) where N is number of images and D is feature dimension
    """
    features_list = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting image features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []

        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                if is_hf:
                    # HuggingFace format: preprocess returns tensor directly
                    image_tensor = preprocess(image)
                    batch_images.append(image_tensor)
                else:
                    # Standard CLIP: preprocess returns tensor, need to unsqueeze
                    image_tensor = preprocess(image).unsqueeze(0)
                    batch_images.append(image_tensor)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue

        if not batch_images:
            continue

        if is_hf:
            # HuggingFace format: already batched by processor
            batch_tensor = torch.cat(batch_images, dim=0).to(device)
            with torch.no_grad():
                batch_features = model.get_image_features(pixel_values=batch_tensor)
        else:
            # Standard CLIP format
            batch_tensor = torch.cat(batch_images, dim=0).to(device)
            with torch.no_grad():
                batch_features = model.encode_image(batch_tensor)

        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        features_list.append(batch_features.cpu())

    if not features_list:
        raise ValueError("No valid images found")

    all_features = torch.cat(features_list, dim=0)
    return all_features


def extract_text_features(
    texts: List[str],
    model,
    device: str = "cuda",
    batch_size: int = 32,
    processor=None,
    is_hf: bool = False
) -> torch.Tensor:
    """
    Extract CLIP text features from a list of text prompts.

    Args:
        texts: List of text prompts
        model: CLIP model
        device: Device to run computation
        batch_size: Batch size
        processor: HuggingFace processor (required if is_hf=True)
        is_hf: Whether using HuggingFace format model

    Returns:
        Tensor of shape (N, D) where N is number of texts and D is feature dimension
    """
    features_list = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting text features"):
        batch_texts = texts[i:i+batch_size]

        if is_hf:
            # HuggingFace format
            if processor is None:
                raise ValueError("processor is required for HuggingFace format")
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                batch_features = model.get_text_features(**inputs)
        else:
            # Standard CLIP format
            text_tokens = clip.tokenize(batch_texts).to(device)
            with torch.no_grad():
                batch_features = model.encode_text(text_tokens)

        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        features_list.append(batch_features.cpu())

    if not features_list:
        raise ValueError("No valid texts found")

    all_features = torch.cat(features_list, dim=0)
    return all_features


def compute_clip_score(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    model,
    device: str = "cuda",
    is_hf: bool = False
) -> Tuple[float, List[float]]:
    """
    Compute CLIP-Score: cosine similarity between image and text features.

    Args:
        image_features: (N, D) tensor of image features
        text_features: (N, D) tensor of text features (should match images)
        model: CLIP model (for logit_scale)
        device: Device to run computation
        is_hf: Whether using HuggingFace format model

    Returns:
        Mean CLIP-Score and list of individual scores
    """
    # Ensure features are on same device
    image_features = image_features.to(device)
    text_features = text_features.to(device)

    # Compute cosine similarity
    # CLIP-Score = logit_scale * cosine_similarity
    if is_hf:
        # HuggingFace format: may not have logit_scale, use default
        if hasattr(model, 'logit_scale') and model.logit_scale is not None:
            logit_scale = model.logit_scale.exp().item()
        else:
            logit_scale = 100.0  # CLIP default logit scale
    else:
        # Standard CLIP format
        logit_scale = model.logit_scale.exp().item()

    # Compute pairwise similarities (assuming aligned pairs)
    similarities = (image_features * text_features).sum(dim=1) * logit_scale

    scores = similarities.cpu().numpy().tolist()
    mean_score = np.mean(scores)

    return mean_score, scores


def compute_cmmd_score(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    device: str = "cuda"
) -> float:
    """
    Compute CMMD score between real and generated image features.

    Args:
        real_features: (N, D) tensor of real image features
        gen_features: (M, D) tensor of generated image features
        device: Device to run computation

    Returns:
        CMMD score (lower is better)
    """
    if not CMMD_AVAILABLE:
        raise ImportError("cmmd-pytorch not available. Please install: pip install cmmd-pytorch")

    real_features = real_features.to(device)
    gen_features = gen_features.to(device)

    # Compute CMMD
    cmmd_score = compute_cmmd(real_features, gen_features)

    return cmmd_score.item()


def evaluate_directory(
    output_dir: str,
    prompts_file: str,
    reference_dir: Optional[str] = None,
    clip_model_name: str = "ViT-L/14",
    device: str = "cuda",
    batch_size: int = 32
) -> Dict:
    """
    Evaluate images in output directory using CMMD and CLIP-Score.

    Args:
        output_dir: Directory containing generated images
        prompts_file: Path to JSON file with prompts
        reference_dir: Optional directory with reference/real images for CMMD
        clip_model_name: CLIP model to use
        device: Device to run computation
        batch_size: Batch size for feature extraction

    Returns:
        Dictionary with evaluation results
    """
    print(f"📊 Evaluating images in: {output_dir}")
    print(f"📝 Using prompts from: {prompts_file}")

    # Load prompts
    prompts_dict = load_prompts(prompts_file)
    print(f"✅ Loaded {len(prompts_dict)} prompts")

    # Collect images
    image_files = collect_images(output_dir)
    print(f"✅ Found {len(image_files)} images")

    if not image_files:
        raise ValueError(f"No images found in {output_dir}")

    # Parse image metadata and match with prompts
    image_metadata = []
    matched_prompts = []
    matched_images = []

    for img_path in image_files:
        metadata = parse_image_filename(img_path)
        if metadata and 'prompt_id' in metadata:
            prompt_id = metadata['prompt_id']
            if prompt_id in prompts_dict:
                image_metadata.append(metadata)
                matched_prompts.append(prompts_dict[prompt_id])
                matched_images.append(img_path)

    print(f"✅ Matched {len(matched_images)} images with prompts")

    if not matched_images:
        raise ValueError("No images matched with prompts. Check filename format.")

    # Load CLIP model
    print(f"🔄 Loading CLIP model: {clip_model_name}")
    model, preprocess, processor, is_hf = load_clip_model(clip_model_name, device=device)

    # Extract features
    print("🔄 Extracting image features...")
    image_features = extract_image_features(matched_images, model, preprocess, device, batch_size, is_hf=is_hf)

    print("🔄 Extracting text features...")
    text_features = extract_text_features(matched_prompts, model, device, batch_size, processor=processor, is_hf=is_hf)

    # Compute CLIP-Score
    print("🔄 Computing CLIP-Score...")
    clip_score_mean, clip_scores = compute_clip_score(image_features, text_features, model, device, is_hf=is_hf)

    results = {
        'output_dir': output_dir,
        'num_images': len(matched_images),
        'clip_score': {
            'mean': float(clip_score_mean),
            'std': float(np.std(clip_scores)),
            'min': float(np.min(clip_scores)),
            'max': float(np.max(clip_scores)),
            'scores': [float(s) for s in clip_scores]
        }
    }

    # Compute CMMD if reference directory provided
    if reference_dir and os.path.exists(reference_dir):
        if not CMMD_AVAILABLE:
            print("⚠️  CMMD library not available. Skipping CMMD computation.")
            print("   To enable CMMD, try installing:")
            print("     pip install git+https://github.com/google-research/cmmd.git")
            print("   Or search for 'cmmd-pytorch' on GitHub for alternative implementations")
            print("   Note: CLIP-Score evaluation completed successfully above.")
        else:
            print(f"🔄 Computing CMMD with reference images from: {reference_dir}")
            reference_images = collect_images(reference_dir)

            if reference_images:
                print(f"✅ Found {len(reference_images)} reference images")
                try:
                    reference_features = extract_image_features(reference_images, model, preprocess, device, batch_size, is_hf=is_hf)
                    cmmd_score = compute_cmmd_score(reference_features, image_features, device)
                    results['cmmd'] = {
                        'score': float(cmmd_score),
                        'reference_dir': reference_dir,
                        'num_reference_images': len(reference_images)
                    }
                except Exception as e:
                    print(f"⚠️  Error computing CMMD: {e}")
                    print("   Skipping CMMD computation, but CLIP-Score is still available")
            else:
                print("⚠️  No reference images found, skipping CMMD")
    elif reference_dir:
        print(f"⚠️  Reference directory does not exist: {reference_dir}")
        print("   Skipping CMMD computation")
    else:
        print("ℹ️  No reference directory provided, skipping CMMD")
        print("   (CMMD requires reference/real images for comparison)")

    return results


def find_image_directories(root_dir: str) -> List[str]:
    """
    Find all directories containing images in the given root directory.

    Args:
        root_dir: Root directory to search

    Returns:
        List of directory paths containing images
    """
    image_dirs = []
    root_path = Path(root_dir)

    # Walk through all subdirectories
    for path in root_path.rglob('*'):
        if path.is_dir():
            # Check if this directory contains images
            has_images = False
            for ext in IMAGE_EXTENSIONS:
                if list(path.glob(f'*.{ext}')) or list(path.glob(f'*.{ext.upper()}')):
                    has_images = True
                    break

            if has_images:
                image_dirs.append(str(path))

    # Sort by path for consistent ordering
    image_dirs.sort()
    return image_dirs


def batch_evaluate_clip_score(
    root_dir: str,
    prompts_file: str,
    clip_model_name: str = "ViT-L/14",
    device: str = "cuda",
    batch_size: int = 32,
    output_json: Optional[str] = None
) -> Dict:
    """
    Batch evaluate CLIP-Score for all image directories under root_dir.

    Args:
        root_dir: Root directory containing subdirectories with images
        prompts_file: Path to JSON file with prompts
        clip_model_name: CLIP model to use
        device: Device to run computation
        batch_size: Batch size for feature extraction
        output_json: Optional path to save results as JSON file

    Returns:
        Dictionary with batch evaluation results
    """
    print("="*80)
    print("🚀 BATCH CLIP-SCORE EVALUATION")
    print("="*80)
    print(f"📁 Root directory: {root_dir}")
    print(f"📝 Prompts file: {prompts_file}")
    print(f"🤖 CLIP model: {clip_model_name}")
    print()

    # Find all directories with images
    print("🔍 Searching for image directories...")
    image_dirs = find_image_directories(root_dir)
    print(f"✅ Found {len(image_dirs)} directories with images")
    print()

    if not image_dirs:
        raise ValueError(f"No image directories found in {root_dir}")

    # Load prompts once
    prompts_dict = load_prompts(prompts_file)
    print(f"✅ Loaded {len(prompts_dict)} prompts")
    print()

    # Load CLIP model once (reuse for all directories)
    print(f"🔄 Loading CLIP model: {clip_model_name}")
    model, preprocess, processor, is_hf = load_clip_model(clip_model_name, device=device)
    print()

    # Evaluate each directory
    all_results = {}
    summary_results = []

    for idx, img_dir in enumerate(image_dirs, 1):
        print("="*80)
        print(f"[{idx}/{len(image_dirs)}] Evaluating: {img_dir}")
        print("="*80)

        try:
            # Collect images
            image_files = collect_images(img_dir)
            if not image_files:
                print(f"⚠️  No images found, skipping...")
                continue

            # Parse image metadata and match with prompts
            matched_prompts = []
            matched_images = []

            for img_path in image_files:
                metadata = parse_image_filename(img_path)
                if metadata and 'prompt_id' in metadata:
                    prompt_id = metadata['prompt_id']
                    if prompt_id in prompts_dict:
                        matched_prompts.append(prompts_dict[prompt_id])
                        matched_images.append(img_path)

            if not matched_images:
                print(f"⚠️  No images matched with prompts, skipping...")
                continue

            print(f"✅ Found {len(matched_images)} matched images")

            # Extract features
            print("🔄 Extracting image features...")
            image_features = extract_image_features(matched_images, model, preprocess, device, batch_size, is_hf=is_hf)

            print("🔄 Extracting text features...")
            text_features = extract_text_features(matched_prompts, model, device, batch_size, processor=processor, is_hf=is_hf)

            # Compute CLIP-Score
            print("🔄 Computing CLIP-Score...")
            clip_score_mean, clip_scores = compute_clip_score(image_features, text_features, model, device, is_hf=is_hf)

            # Store results
            result = {
                'directory': img_dir,
                'num_images': len(matched_images),
                'clip_score': {
                    'mean': float(clip_score_mean),
                    'std': float(np.std(clip_scores)),
                    'min': float(np.min(clip_scores)),
                    'max': float(np.max(clip_scores))
                }
            }

            all_results[img_dir] = result

            # Extract metadata for summary
            relative_path = os.path.relpath(img_dir, root_dir)
            path_parts = relative_path.split(os.sep)

            summary_entry = {
                'path': relative_path,
                'steps': None,
                'sampler': None,
                'clip_score_mean': float(clip_score_mean),
                'clip_score_std': float(np.std(clip_scores)),
                'num_images': len(matched_images)
            }

            # Try to extract steps and sampler from path
            for part in path_parts:
                if part.startswith('steps_'):
                    try:
                        summary_entry['steps'] = int(part.replace('steps_', ''))
                    except:
                        pass
                elif part in ['dpm_lm', 'dpm++', 'unipc', 'ddim', 'pndm']:
                    summary_entry['sampler'] = part

            summary_results.append(summary_entry)

            print(f"✅ CLIP-Score: {clip_score_mean:.4f} ± {np.std(clip_scores):.4f}")
            print()

        except Exception as e:
            print(f"❌ Error evaluating {img_dir}: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue

    # Create summary
    batch_results = {
        'root_dir': root_dir,
        'prompts_file': prompts_file,
        'clip_model': clip_model_name,
        'total_directories': len(image_dirs),
        'evaluated_directories': len(all_results),
        'results': all_results,
        'summary': summary_results
    }

    # Print summary table
    print("="*80)
    print("📊 BATCH EVALUATION SUMMARY")
    print("="*80)

    if summary_results:
        # Sort by steps and sampler for better readability
        summary_sorted = sorted(summary_results, key=lambda x: (
            x['steps'] if x['steps'] is not None else 999,
            x['sampler'] or ''
        ))

        print(f"\n{'Path':<50} {'Steps':<8} {'Sampler':<10} {'CLIP-Score':<15} {'Images':<8}")
        print("-" * 100)
        for entry in summary_sorted:
            steps_str = str(entry['steps']) if entry['steps'] is not None else 'N/A'
            sampler_str = entry['sampler'] or 'N/A'
            score_str = f"{entry['clip_score_mean']:.4f} ± {entry['clip_score_std']:.4f}"
            path_str = entry['path'][:48]  # Truncate if too long
            print(f"{path_str:<50} {steps_str:<8} {sampler_str:<10} {score_str:<15} {entry['num_images']:<8}")

        # Statistics
        all_means = [r['clip_score_mean'] for r in summary_results]
        print("\n" + "-" * 100)
        print(f"Overall Statistics:")
        print(f"  Mean CLIP-Score: {np.mean(all_means):.4f} ± {np.std(all_means):.4f}")
        print(f"  Min CLIP-Score:  {np.min(all_means):.4f}")
        print(f"  Max CLIP-Score:  {np.max(all_means):.4f}")

    # Save results if requested
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {output_json}")

    return batch_results


def batch_evaluate_all_metrics(
    root_dir: str,
    prompts_file: str,
    reference_dir: Optional[str] = None,
    clip_model_name: str = "ViT-L/14",
    device: str = "cuda",
    batch_size: int = 32,
    output_json: Optional[str] = None
) -> Dict:
    """
    Batch evaluate both CLIP-Score and CMMD for all image directories under root_dir.

    This function evaluates all subdirectories and computes both metrics when possible.

    Args:
        root_dir: Root directory containing subdirectories with images
        prompts_file: Path to JSON file with prompts
        reference_dir: Optional directory with reference/real images for CMMD
        clip_model_name: CLIP model to use
        device: Device to run computation
        batch_size: Batch size for feature extraction
        output_json: Optional path to save results as JSON file

    Returns:
        Dictionary with batch evaluation results including both CLIP-Score and CMMD
    """
    print("="*80)
    print("🚀 BATCH EVALUATION (CLIP-Score + CMMD)")
    print("="*80)
    print(f"📁 Root directory: {root_dir}")
    print(f"📝 Prompts file: {prompts_file}")
    print(f"🤖 CLIP model: {clip_model_name}")
    if reference_dir:
        print(f"📊 Reference directory: {reference_dir}")
    print()

    # Find all directories with images
    print("🔍 Searching for image directories...")
    image_dirs = find_image_directories(root_dir)
    print(f"✅ Found {len(image_dirs)} directories with images")
    print()

    if not image_dirs:
        raise ValueError(f"No image directories found in {root_dir}")

    # Load prompts once
    prompts_dict = load_prompts(prompts_file)
    print(f"✅ Loaded {len(prompts_dict)} prompts")
    print()

    # Load CLIP model once (reuse for all directories)
    print(f"🔄 Loading CLIP model: {clip_model_name}")
    model, preprocess, processor, is_hf = load_clip_model(clip_model_name, device=device)
    print()

    # Load reference images once if provided (for CMMD)
    reference_features = None
    reference_images = None
    if reference_dir and os.path.exists(reference_dir):
        if CMMD_AVAILABLE:
            print(f"🔄 Loading reference images for CMMD from: {reference_dir}")
            reference_images = collect_images(reference_dir)
            if reference_images:
                print(f"✅ Found {len(reference_images)} reference images")
                reference_features = extract_image_features(reference_images, model, preprocess, device, batch_size, is_hf=is_hf)
                print("✅ Reference features extracted")
            else:
                print("⚠️  No reference images found, CMMD will be skipped")
        else:
            print("⚠️  CMMD library not available. Only CLIP-Score will be computed.")
            print("   To enable CMMD, try: pip install git+https://github.com/google-research/cmmd.git")
        print()
    elif reference_dir:
        print(f"⚠️  Reference directory does not exist: {reference_dir}")
        print("   CMMD will be skipped")
        print()

    # Evaluate each directory
    all_results = {}
    summary_results = []

    for idx, img_dir in enumerate(image_dirs, 1):
        print("="*80)
        print(f"[{idx}/{len(image_dirs)}] Evaluating: {img_dir}")
        print("="*80)

        try:
            # Collect images
            image_files = collect_images(img_dir)
            if not image_files:
                print(f"⚠️  No images found, skipping...")
                continue

            # Parse image metadata and match with prompts
            matched_prompts = []
            matched_images = []

            for img_path in image_files:
                metadata = parse_image_filename(img_path)
                if metadata and 'prompt_id' in metadata:
                    prompt_id = metadata['prompt_id']
                    if prompt_id in prompts_dict:
                        matched_prompts.append(prompts_dict[prompt_id])
                        matched_images.append(img_path)

            if not matched_images:
                print(f"⚠️  No images matched with prompts, skipping...")
                continue

            print(f"✅ Found {len(matched_images)} matched images")

            # Extract features
            print("🔄 Extracting image features...")
            image_features = extract_image_features(matched_images, model, preprocess, device, batch_size, is_hf=is_hf)

            print("🔄 Extracting text features...")
            text_features = extract_text_features(matched_prompts, model, device, batch_size, processor=processor, is_hf=is_hf)

            # Compute CLIP-Score
            print("🔄 Computing CLIP-Score...")
            clip_score_mean, clip_scores = compute_clip_score(image_features, text_features, model, device, is_hf=is_hf)

            # Store results
            result = {
                'directory': img_dir,
                'num_images': len(matched_images),
                'clip_score': {
                    'mean': float(clip_score_mean),
                    'std': float(np.std(clip_scores)),
                    'min': float(np.min(clip_scores)),
                    'max': float(np.max(clip_scores))
                }
            }

            # Compute CMMD if reference features available
            if reference_features is not None and CMMD_AVAILABLE:
                print("🔄 Computing CMMD...")
                try:
                    cmmd_score = compute_cmmd_score(reference_features, image_features, device)
                    result['cmmd'] = {
                        'score': float(cmmd_score),
                        'reference_dir': reference_dir,
                        'num_reference_images': len(reference_images) if reference_images is not None else 0
                    }
                    print(f"✅ CMMD Score: {cmmd_score:.4f}")
                except Exception as e:
                    print(f"⚠️  Error computing CMMD: {e}")
                    print("   CMMD computation failed, but CLIP-Score is available")

            all_results[img_dir] = result

            # Extract metadata for summary
            relative_path = os.path.relpath(img_dir, root_dir)
            path_parts = relative_path.split(os.sep)

            summary_entry = {
                'path': relative_path,
                'steps': None,
                'sampler': None,
                'clip_score_mean': float(clip_score_mean),
                'clip_score_std': float(np.std(clip_scores)),
                'num_images': len(matched_images)
            }

            # Add CMMD to summary if available
            if 'cmmd' in result:
                summary_entry['cmmd_score'] = result['cmmd']['score']

            # Try to extract steps and sampler from path
            for part in path_parts:
                if part.startswith('steps_'):
                    try:
                        summary_entry['steps'] = int(part.replace('steps_', ''))
                    except:
                        pass
                elif part in ['dpm_lm', 'dpm++', 'unipc', 'ddim', 'pndm', 'dpm_hcg']:
                    summary_entry['sampler'] = part

            summary_results.append(summary_entry)

            print(f"✅ CLIP-Score: {clip_score_mean:.4f} ± {np.std(clip_scores):.4f}")
            if 'cmmd' in result:
                print(f"✅ CMMD Score: {result['cmmd']['score']:.4f}")
            print()

        except Exception as e:
            print(f"❌ Error evaluating {img_dir}: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue

    # Create summary
    batch_results = {
        'root_dir': root_dir,
        'prompts_file': prompts_file,
        'clip_model': clip_model_name,
        'reference_dir': reference_dir,
        'total_directories': len(image_dirs),
        'evaluated_directories': len(all_results),
        'results': all_results,
        'summary': summary_results
    }

    # Print summary table
    print("="*80)
    print("📊 BATCH EVALUATION SUMMARY")
    print("="*80)

    if summary_results:
        # Sort by steps and sampler for better readability
        summary_sorted = sorted(summary_results, key=lambda x: (
            x['steps'] if x['steps'] is not None else 999,
            x['sampler'] or ''
        ))

        # Check if CMMD is available
        has_cmmd = any('cmmd_score' in entry for entry in summary_results)

        if has_cmmd:
            print(f"\n{'Path':<45} {'Steps':<8} {'Sampler':<10} {'CLIP-Score':<18} {'CMMD':<12} {'Images':<8}")
            print("-" * 110)
            for entry in summary_sorted:
                steps_str = str(entry['steps']) if entry['steps'] is not None else 'N/A'
                sampler_str = entry['sampler'] or 'N/A'
                score_str = f"{entry['clip_score_mean']:.4f} ± {entry['clip_score_std']:.4f}"
                cmmd_str = f"{entry.get('cmmd_score', 'N/A'):.4f}" if 'cmmd_score' in entry else 'N/A'
                path_str = entry['path'][:43]  # Truncate if too long
                print(f"{path_str:<45} {steps_str:<8} {sampler_str:<10} {score_str:<18} {cmmd_str:<12} {entry['num_images']:<8}")
        else:
            print(f"\n{'Path':<50} {'Steps':<8} {'Sampler':<10} {'CLIP-Score':<15} {'Images':<8}")
            print("-" * 100)
            for entry in summary_sorted:
                steps_str = str(entry['steps']) if entry['steps'] is not None else 'N/A'
                sampler_str = entry['sampler'] or 'N/A'
                score_str = f"{entry['clip_score_mean']:.4f} ± {entry['clip_score_std']:.4f}"
                path_str = entry['path'][:48]  # Truncate if too long
                print(f"{path_str:<50} {steps_str:<8} {sampler_str:<10} {score_str:<15} {entry['num_images']:<8}")

        # Statistics
        all_means = [r['clip_score_mean'] for r in summary_results]
        print("\n" + "-" * 100)
        print(f"Overall Statistics (CLIP-Score):")
        print(f"  Mean: {np.mean(all_means):.4f} ± {np.std(all_means):.4f}")
        print(f"  Min:  {np.min(all_means):.4f}")
        print(f"  Max:  {np.max(all_means):.4f}")

        if has_cmmd:
            cmmd_scores = [r.get('cmmd_score') for r in summary_results if 'cmmd_score' in r]
            if cmmd_scores:
                print(f"\nOverall Statistics (CMMD):")
                print(f"  Mean: {np.mean(cmmd_scores):.4f} ± {np.std(cmmd_scores):.4f}")
                print(f"  Min:  {np.min(cmmd_scores):.4f}")
                print(f"  Max:  {np.max(cmmd_scores):.4f}")

    # Save results if requested
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {output_json}")

    return batch_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated images using CMMD and CLIP-Score metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/',
        help='Directory containing generated images'
    )

    parser.add_argument(
        '--prompts_file',
        type=str,
        default='evaluations/coco_prompts/coco_top_40_prompts_backup.json',
        help='Path to JSON file containing prompts'
    )

    parser.add_argument(
        '--reference_dir',
        type=str,
        default='/home/liying/Documents/dataset/coco/val2014',
        help='Directory containing reference/real images for CMMD computation (default: /home/liying/Documents/dataset/coco/val2014)'
    )

    parser.add_argument(
        '--clip_model',
        type=str,
        default='/home/liying/Documents/clip-vit-large-patch14',
        help='CLIP model to use for feature extraction. Can be either:\n'
             '  - Standard CLIP model name (e.g., "ViT-L/14", "ViT-B/32")\n'
             '  - Local path to HuggingFace format model (e.g., "/path/to/clip-vit-large-patch14")'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run computation'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for feature extraction'
    )

    parser.add_argument(
        '--output_json',
        type=str,
        default=None,
        help='Optional: Path to save results as JSON file'
    )

    parser.add_argument(
        '--batch_mode',
        action='store_true',
        help='Enable batch mode: evaluate all subdirectories under output_dir (CLIP-Score only)'
    )

    parser.add_argument(
        '--batch_all_metrics',
        action='store_true',
        help='Enable batch mode with all metrics: evaluate all subdirectories with both CLIP-Score and CMMD'
    )

    args = parser.parse_args()

    # Check if CLIP is available
    if not CLIP_AVAILABLE:
        print("❌ Error: CLIP library not available.")
        print("   Please install: pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)

    # Run evaluation
    try:
        if args.batch_all_metrics:
            # Batch mode with all metrics: evaluate all subdirectories with CLIP-Score and CMMD
            results = batch_evaluate_all_metrics(
                root_dir=args.output_dir,
                prompts_file=args.prompts_file,
                reference_dir=args.reference_dir,
                clip_model_name=args.clip_model,
                device=args.device,
                batch_size=args.batch_size,
                output_json=args.output_json
            )
        elif args.batch_mode:
            # Batch mode: evaluate all subdirectories (CLIP-Score only)
            results = batch_evaluate_clip_score(
                root_dir=args.output_dir,
                prompts_file=args.prompts_file,
                clip_model_name=args.clip_model,
                device=args.device,
                batch_size=args.batch_size,
                output_json=args.output_json
            )
        else:
            # Single directory mode
            results = evaluate_directory(
                output_dir=args.output_dir,
                prompts_file=args.prompts_file,
                reference_dir=args.reference_dir,
                clip_model_name=args.clip_model,
                device=args.device,
                batch_size=args.batch_size
            )

            # Print results
            print("\n" + "="*60)
            print("📊 EVALUATION RESULTS")
            print("="*60)
            print(f"Number of images evaluated: {results['num_images']}")
            print(f"\n🎯 CLIP-Score:")
            print(f"   Mean: {results['clip_score']['mean']:.4f}")
            print(f"   Std:  {results['clip_score']['std']:.4f}")
            print(f"   Min:  {results['clip_score']['min']:.4f}")
            print(f"   Max:  {results['clip_score']['max']:.4f}")

            if 'cmmd' in results:
                print(f"\n📏 CMMD Score:")
                print(f"   Score: {results['cmmd']['score']:.4f}")
                print(f"   Reference images: {results['cmmd']['num_reference_images']}")

            # Save results if requested
            if args.output_json:
                with open(args.output_json, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Results saved to: {args.output_json}")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
