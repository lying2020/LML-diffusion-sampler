# CMMD and CLIP-Score Evaluation

This script evaluates generated images using two metrics:
1. **CMMD (Conditional Maximum Mean Discrepancy)** - Measures the distribution distance between real and generated images
2. **CLIP-Score** - Measures the alignment between images and their corresponding text prompts

## Installation

Before using this script, install the required dependencies:

```bash
# Install CLIP
pip install git+https://github.com/openai/CLIP.git

# Install CMMD library (optional, for CMMD metric)
# Option 1: Try from PyPI (may not be available)
pip install cmmd-pytorch

# Option 2: Install from GitHub (if PyPI doesn't work)
# pip install git+https://github.com/google-research/cmmd.git

# Note: If CMMD installation fails, CLIP-Score evaluation will still work
```

## Usage

### Basic Usage (Single Directory)

```bash
python evaluations/cmmd_clipscore_eval.py \
    --output_dir output/coco_stable-diffusion-2-base/steps_20/dpm_lm \
    --prompts_file evaluations/coco_prompts/coco_top_40_prompts_backup.json
```

### Batch Mode (Evaluate All Subdirectories)

#### Option 1: CLIP-Score Only

To evaluate all subdirectories with CLIP-Score only:

```bash
python evaluations/cmmd_clipscore_eval.py \
    --output_dir output/coco_stable-diffusion-2-base \
    --prompts_file evaluations/coco_prompts/coco_top_40_prompts_backup.json \
    --batch_mode \
    --output_json batch_clip_score_results.json
```

#### Option 2: CLIP-Score + CMMD (Recommended)

To evaluate all subdirectories with both CLIP-Score and CMMD:

```bash
python evaluations/cmmd_clipscore_eval.py \
    --output_dir output/coco_stable-diffusion-2-base \
    --prompts_file evaluations/coco_prompts/coco_top_40_prompts_backup.json \
    --batch_all_metrics \
    --output_json batch_all_metrics_results.json
```

This will:
- Automatically find all subdirectories containing images
- Evaluate CLIP-Score for each directory
- Evaluate CMMD for each directory (if reference images and CMMD library available)
- Generate a comprehensive summary table with both metrics
- Save all results to a JSON file

**Example for your specific path**:
```bash
python evaluations/cmmd_clipscore_eval.py \
    --output_dir output/coco_stable-diffusion-2-base \
    --prompts_file evaluations/coco_prompts/coco_top_40_prompts_backup.json \
    --batch_all_metrics \
    --output_json output/coco_stable-diffusion-2-base/evaluation_results.json
```

### With CMMD (requires reference images)

```bash
python evaluations/cmmd_clipscore_eval.py \
    --output_dir output/coco_stable-diffusion-2-base/steps_20/dpm_lm \
    --prompts_file evaluations/coco_prompts/coco_top_40_prompts_backup.json \
    --reference_dir /home/liying/Documents/dataset/coco/val2014
```

**Note**: The default reference directory is set to `/home/liying/Documents/dataset/coco/val2014`. If you want to use a different directory, specify it with `--reference_dir`.

**Important**: CMMD requires reference/real images for comparison. See [CMMD_REFERENCE_DATASET_GUIDE.md](CMMD_REFERENCE_DATASET_GUIDE.md) for details on preparing reference datasets.

### Using Local CLIP Model

If you have a local HuggingFace format CLIP model, you can specify the path:

```bash
python evaluations/cmmd_clipscore_eval.py \
    --output_dir output/coco_stable-diffusion-2-base/steps_20/dpm_lm \
    --prompts_file evaluations/coco_prompts/coco_top_40_prompts_backup.json \
    --clip_model /home/liying/Documents/clip-vit-large-patch14
```

### Advanced Options

```bash
python evaluations/cmmd_clipscore_eval.py \
    --output_dir output/coco_stable-diffusion-2-base/steps_20/dpm_lm \
    --prompts_file evaluations/coco_prompts/coco_top_40_prompts_backup.json \
    --clip_model ViT-L/14 \
    --device cuda \
    --batch_size 32 \
    --output_json results.json
```

## Arguments

- `--output_dir`: Directory containing generated images (default: `output/`)
- `--prompts_file`: Path to JSON file with prompts (default: `evaluations/coco_prompts/coco_top_40_prompts_backup.json`)
- `--reference_dir`: Optional directory with reference/real images for CMMD computation
- `--clip_model`: CLIP model to use (default: `ViT-L/14`)
  - Can be either:
    - Standard CLIP model name: `ViT-B/32`, `ViT-B/16`, `ViT-L/14`, `ViT-L/14@336px`, `RN50`, `RN101`, `RN50x4`, `RN50x16`, `RN50x64`
    - Local path to HuggingFace format model: `/path/to/clip-vit-large-patch14`
- `--device`: Device to run computation (default: `cuda` if available, else `cpu`)
- `--batch_size`: Batch size for feature extraction (default: `32`)
- `--output_json`: Optional path to save results as JSON file
- `--batch_mode`: Enable batch mode to evaluate all subdirectories under output_dir (CLIP-Score only)
- `--batch_all_metrics`: Enable batch mode with all metrics (CLIP-Score + CMMD) for all subdirectories

## Image Filename Format

The script expects image filenames in the following format:
```
{index:05d}_{prompt_id}_{guidance}_{inference_steps}_{seed}_{sampler}.jpg
```

Example:
```
00000_1_guidance7.5_inference10_seed6_dpm_lm.jpg
```

Where:
- `00000` is the image index
- `1` is the prompt ID (must match a key in the prompts JSON file)
- `guidance7.5` is the guidance scale
- `inference10` is the number of inference steps
- `seed6` is the random seed
- `dpm_lm` is the sampler name

## Prompts JSON Format

The prompts file should be a JSON dictionary mapping prompt IDs to text prompts:

```json
{
  "1": "Running action: a professional athlete sprinting on a track...",
  "2": "Tennis serve: a tennis player executing a powerful serve...",
  ...
}
```

## Output

The script prints evaluation results to the console:

```
📊 EVALUATION RESULTS
============================================================
Number of images evaluated: 20

🎯 CLIP-Score:
   Mean: 0.8234
   Std:  0.0456
   Min:  0.7123
   Max:  0.9234

📏 CMMD Score:
   Score: 0.1234
   Reference images: 1000
```

If `--output_json` is specified, results are also saved to a JSON file.

## Notes

- **CLIP-Score**: Always computed if images and prompts are matched successfully
- **CMMD**: Only computed if `--reference_dir` is provided with reference images
  - **Reference dataset**: Should be real images from the same domain as your generated images
  - **For COCO prompts**: Use MS COCO validation set (see [CMMD_REFERENCE_DATASET_GUIDE.md](CMMD_REFERENCE_DATASET_GUIDE.md))
  - **Image count**: More reference images (1000+) give more stable CMMD scores
- The script automatically matches images to prompts based on the prompt ID in the filename
- Images that don't match any prompt are skipped

## About CMMD Reference Dataset

**Stable Diffusion 2.0** is trained on **LAION-5B**, not MS COCO. However, for CMMD evaluation with COCO prompts:

✅ **Recommended**: Use MS COCO 2017 validation set as reference
- Download: http://images.cocodataset.org/zips/val2017.zip
- Contains ~5,000 high-quality images
- Matches your generation domain

See [CMMD_REFERENCE_DATASET_GUIDE.md](CMMD_REFERENCE_DATASET_GUIDE.md) for detailed information.

## Troubleshooting

1. **No images matched with prompts**: Check that image filenames contain prompt IDs that exist in the prompts JSON file
2. **CMMD not computed**: Provide `--reference_dir` with reference images
3. **CUDA out of memory**: Reduce `--batch_size` or use CPU with `--device cpu`
