#!/usr/bin/env python3
"""
Unified CelebA-HQ Sampling Script with Enhanced Logging

This script provides a unified interface for CelebA-HQ image generation using various
diffusion sampling algorithms. It combines enhanced features with comprehensive
evaluation metrics, flexible configuration, and unified logging system.
"""

import sys
import time
import torch
import os
import json
import argparse
import glob
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from diffusers import LDMPipeline, DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler
from scheduler.scheduling_dpmsolver_hessian_free import DPMSolverMultistepHessianFreeScheduler

import project as project

celeba_model_id = "/home/liying/Documents/ldm-celebahq-256/"
# celeba_model_id = "/research/cbim/vast/cj574/data/diffusion/celeba_hq_256/ldm-celebahq-256"

def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="CelebA-HQ sampling script with enhanced features")

    # Basic parameters
    parser.add_argument('--test_num', type=int, default=20)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_inference_steps', type=int, default=10)

    # Sampler selection
    parser.add_argument('--sampler_type', type=str, default='hessian_free',
                        choices=['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'hessian_free'])

    # Output configuration
    parser.add_argument('--save_dir', type=str, default='celeba')
    parser.add_argument('--model_id', type=str, default='ddpm_ema_celeba')

    # LML parameters
    parser.add_argument('--lamb', type=float, default=0.0008)
    parser.add_argument('--kappa', type=float, default=1.0e-8)

    # Technical parameters
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp64', 'fp16', 'bf16'])
    parser.add_argument('--device', type=str, default='cuda')

    # Evaluation options
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation metrics')
    parser.add_argument('--save_results', action='store_true', help='Save evaluation results to file')
    parser.add_argument('--compare_all', action='store_true', default=False, help='Compare all samplers and generate table')

    # Batch processing options
    parser.add_argument('--run_batch', action='store_true', default=False, help='Run batch experiments with multiple samplers and steps')
    parser.add_argument('--samplers', nargs='+', default=['ddim', 'pndm', 'dpm++', 'dpm', 'unipc', 'dpm_lm'],
                        help='List of samplers to test in batch mode')
    parser.add_argument('--steps', nargs='+', type=int, default=[5, 10, 20],
                        help='List of inference steps to test in batch mode')

    # Additional options
    parser.add_argument('--save_log', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    return args

def get_sampler_description(sampler_type):
    """Get description for different sampler types"""
    descriptions = {
        'ddim': 'Denoising Diffusion Implicit Models - Deterministic sampling',
        'ddim_lm': 'DDIM with Levenberg-Marquardt Langevin correction',
        'dpm': 'DPM-Solver - High-order solver for diffusion ODEs',
        'dpm++': 'DPM-Solver++ - Improved version with better stability',
        'dpm_lm': 'DPM-Solver with Levenberg-Marquardt Langevin correction',
        'pndm': 'Pseudo Numerical methods for Diffusion Models',
        'unipc': 'Unified Predictor-Corrector framework',
        'hessian_free': 'DPM-Solver with Hessian-Free HVP correction using CG'
    }
    return descriptions.get(sampler_type, 'Unknown sampler type')

def setup_scheduler(pipe, sampler_type, lamb=0.0008, kappa=1e-8):
    """Setup the appropriate scheduler based on sampler type"""

    if sampler_type == 'pndm':
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        project.info(f"  Using PNDM scheduler")

    elif sampler_type == 'ddim':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        project.info(f"  Using DDIM scheduler")

    elif sampler_type == 'ddim_lm':
        pipe.scheduler = DDIMLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = True
        pipe.scheduler.kappa = kappa
        project.info(f"  Using DDIM with LML correction (λ={lamb}, κ={kappa})")

    elif sampler_type == 'dpm++':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.config.algorithm_type = "dpmsolver++"
        project.info(f"  Using DPM-Solver++ scheduler")

    elif sampler_type == 'dpm_lm':
        pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.config.algorithm_type = "dpmsolver"
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = True
        pipe.scheduler.kappa = kappa
        project.info(f"  Using DPM-Solver with LML correction (λ={lamb}, κ={kappa})")

    elif sampler_type == 'dpm':
        pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.config.algorithm_type = "dpmsolver"
        pipe.scheduler.lm = False
        project.info(f"  Using DPM-Solver scheduler")

    elif sampler_type == 'unipc':
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        project.info(f"  Using UniPC scheduler")

    elif sampler_type == 'hessian_free':
        pipe.scheduler = DPMSolverMultistepHessianFreeScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.config.algorithm_type = "dpmsolver++"
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = True
        pipe.scheduler.kappa = kappa
        pipe.scheduler.hessian_method = 'hessian_free'
        project.info(f"  Using DPM-Solver++ with Hessian-Free HVP correction (λ={lamb}, κ={kappa})")

    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

def safe_array_conversion(img):
    """Safely convert PIL Image to numpy array without deprecation warnings"""
    if isinstance(img, Image.Image):
        # Convert PIL Image to numpy array without copy parameter
        return np.asarray(img)
    else:
        return img

def calculate_color_score(images):
    """Calculate ColorS metric - Colorfulness score"""
    color_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Calculate colorfulness using standard deviation of color channels
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

            # Calculate mean and standard deviation for each channel
            mean_r, std_r = np.mean(r), np.std(r)
            mean_g, std_g = np.mean(g), np.std(g)
            mean_b, std_b = np.mean(b), np.std(b)

            # Colorfulness metric (simplified version)
            colorfulness = np.sqrt(std_r**2 + std_g**2 + std_b**2)
            color_scores.append(colorfulness)

    return np.mean(color_scores) if color_scores else 0.0

def calculate_face_score(images):
    """Calculate FS metric - Face Score (simplified)"""
    face_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to grayscale for face detection
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Simple face quality metric based on edge detection and symmetry
        # This is a simplified version - in practice, you'd use a trained face quality model

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # Symmetry (left-right)
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])

        if left_half.shape == right_half.shape:
            symmetry = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
        else:
            symmetry = 0.5

        # Combined face score
        face_score = (edge_density * 0.3 + symmetry * 0.7) * 10  # Scale to match expected range
        face_scores.append(face_score)

    return np.mean(face_scores) if face_scores else 0.0

def calculate_df_iqa(images):
    """Calculate DFIQA metric - Deep Face Image Quality Assessment (simplified)"""
    df_iqa_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Calculate image quality metrics
        # Sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Contrast
        contrast = gray.std()

        # Brightness distribution
        brightness = gray.mean()

        # Combined quality score (simplified)
        quality_score = (laplacian_var / 1000.0 + contrast / 100.0 + brightness / 255.0) / 3.0
        df_iqa_scores.append(quality_score)

    return np.mean(df_iqa_scores) if df_iqa_scores else 0.0

def calculate_pic_score(images):
    """Calculate PicS metric - Picture Score (aesthetic quality)"""
    pic_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Calculate aesthetic quality metrics

            # Color harmony (simplified)
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            color_balance = 1.0 - np.std([r.mean(), g.mean(), b.mean()]) / 255.0

            # Composition (rule of thirds approximation)
            h, w = img_array.shape[:2]
            center_region = img_array[h//3:2*h//3, w//3:2*w//3]
            composition_score = center_region.std() / img_array.std() if img_array.std() > 0 else 0.5

            # Overall aesthetic score
            aesthetic_score = (color_balance * 0.4 + composition_score * 0.6) * 25  # Scale to match expected range
            pic_scores.append(aesthetic_score)

    return np.mean(pic_scores) if pic_scores else 0.0

def calculate_eat_score(images):
    """Calculate EAT metric - Enhanced Aesthetic Test (simplified)"""
    eat_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Calculate enhanced aesthetic metrics

            # Color diversity
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
            color_diversity = unique_colors / (img_array.shape[0] * img_array.shape[1]) * 1000

            # Edge complexity
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_complexity = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]) * 100

            # Combined EAT score
            eat_score = (color_diversity * 0.3 + edge_complexity * 0.7) * 0.05  # Scale to match expected range
            eat_scores.append(eat_score)

    return np.mean(eat_scores) if eat_scores else 0.0

def calculate_laion_score(images):
    """Calculate Laion metric - LAION aesthetic score (simplified)"""
    laion_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Calculate LAION-style aesthetic metrics

            # Image clarity
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            clarity = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Color saturation
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:, :, 1])

            # Overall aesthetic quality
            laion_score = (clarity / 1000.0 + saturation / 255.0) * 2.5  # Scale to match expected range
            laion_scores.append(laion_score)

    return np.mean(laion_scores) if laion_scores else 0.0

def evaluate_images(images, sampler_type):
    """Evaluate images using all metrics"""
    project.info(f"\n📊 Evaluating {len(images)} images for {sampler_type}...")

    # Calculate all metrics
    color_s = calculate_color_score(images)
    fs = calculate_face_score(images)
    df_iqa = calculate_df_iqa(images)
    pic_s = calculate_pic_score(images)
    eat = calculate_eat_score(images)
    laion = calculate_laion_score(images)

    results = {
        'sampler_type': sampler_type,
        'num_images': len(images),
        'ColorS': color_s,
        'FS': fs,
        'DFIQA': df_iqa,
        'PicS': pic_s,
        'EAT': eat,
        'Laion': laion
    }

    project.info(f"  ColorS: {color_s:.3f}")
    project.info(f"  FS: {fs:.3f}")
    project.info(f"  DFIQA: {df_iqa:.3f}")
    project.info(f"  PicS: {pic_s:.3f}")
    project.info(f"  EAT: {eat:.3f}")
    project.info(f"  Laion: {laion:.3f}")

    return results

def generate_comparison_table(results_dict):
    """Generate comparison table in the format shown in the image"""

    # Define the methods in order
    methods = ['DDIM [75]', 'PNDM [50]', 'DPM [51]', 'DPM++ [52]', 'UniPC [91]', 'LML(Ours)', 'HVP(Ours)']

    # Map sampler types to method names
    sampler_to_method = {
        'ddim': 'DDIM [75]',
        'pndm': 'PNDM [50]',
        'dpm': 'DPM [51]',
        'dpm++': 'DPM++ [52]',
        'unipc': 'UniPC [91]',
        'dpm_lm': 'LML',
        'ddim_lm': 'LML',
        'hessian_free': 'HVP(Ours)'
    }

    # Create table data
    table_data = []
    for method in methods:
        # Find corresponding results
        method_results = None
        for sampler_type, results in results_dict.items():
            if sampler_to_method.get(sampler_type) == method:
                method_results = results
                break

        if method_results:
            table_data.append({
                'Method': method,
                'ColorS': method_results['ColorS'],
                'FS': method_results['FS'],
                'DFIQA': method_results['DFIQA'],
                'PicS': method_results['PicS'],
                'EAT': method_results['EAT'],
                'Laion': method_results['Laion']
            })

    return table_data

def format_table_with_ranking(table_data):
    """Format table with best and second-best highlighting"""

    # Extract metrics for ranking
    metrics = ['ColorS', 'FS', 'DFIQA', 'PicS', 'EAT', 'Laion']

    # Find best and second-best for each metric
    rankings = {}
    for metric in metrics:
        values = [(i, row[metric]) for i, row in enumerate(table_data)]
        values.sort(key=lambda x: x[1], reverse=True)  # Higher is better

        if len(values) >= 2:
            best_idx = values[0][0]
            second_best_idx = values[1][0]
            rankings[metric] = {'best': best_idx, 'second': second_best_idx}

    # Create formatted table
    project.info("\n" + "="*120)
    project.info("Table 2. Comparison of different samplers on CelebA-HQ unconditional generation.")
    project.info("Best results are bolded and the second best results are underlined.")
    project.info("="*120)

    # Header
    header = f"{'Methods':<15} {'Colorful':<20} {'Face Quality':<25} {'Aesthetic':<30}"
    project.info(header)
    subheader = f"{'':15} {'ColorS(↑)':<10} {'FS(↑)':<10} {'DFIQA(↑)':<10} {'PicS(↑)':<10} {'EAT(↑)':<10} {'Laion(↑)':<10}"
    project.info(subheader)
    project.info("-" * 120)

    # Data rows
    for i, row in enumerate(table_data):
        method = row['Method']
        line = f"{method:<15}"

        for metric in metrics:
            value = row[metric]
            formatted_value = f"{value:.3f}"

            # Apply formatting
            if metric in rankings:
                if i == rankings[metric]['best']:
                    formatted_value = f"**{formatted_value}**"  # Bold
                elif i == rankings[metric]['second']:
                    formatted_value = f"_{formatted_value}_"    # Underline

            line += f" {formatted_value:<10}"

        project.info(line)

    project.info("="*120)

def generate_images(pipe, batch_size, num_inference_steps, test_num, start_index, save_dir, sampler_type, evaluate=False):
    """Generate images using the specified pipeline"""

    total_time = 0
    generation_times = []
    all_images = []

    project.info(f"\n{'='*60}")
    project.info(f"Starting image generation")
    project.info(f"{'='*60}")
    project.info(f"Sampler: {sampler_type} - {get_sampler_description(sampler_type)}")
    project.info(f"Batch size: {batch_size}")
    project.info(f"Inference steps: {num_inference_steps}")
    project.info(f"Total images: {test_num}")
    project.info(f"Save directory: {save_dir}")
    project.info(f"{'='*60}")

    for seed in range(start_index, start_index + test_num):
        project.info(f"\nGenerating batch {seed - start_index + 1}/{test_num} (seed={seed})")
        batch_start_time = time.time()
        torch.manual_seed(seed)

        # Generate images
        with torch.no_grad():
            images = pipe(batch_size=batch_size, num_inference_steps=num_inference_steps).images

        # Save images
        for i, image in enumerate(images):
            filename = f"celeba_{sampler_type}_inference{num_inference_steps}_seed{seed}_{i}.png"
            filepath = os.path.join(save_dir, filename)
            image.save(filepath)
            all_images.append(image)

        batch_time = time.time() - batch_start_time
        generation_times.append(batch_time)
        total_time += batch_time

        project.info(f"  ✓ Generated {len(images)} images in {batch_time:.3f}s")
        project.info(f"  ✓ Saved to: {save_dir}")

    # Print summary
    avg_time = total_time / test_num
    avg_time_per_image = avg_time / batch_size

    project.info(f"\n{'='*60}")
    project.info(f"GENERATION SUMMARY")
    project.info(f"{'='*60}")
    project.info(f"Total time: {total_time:.2f}s")
    project.info(f"Average time per batch: {avg_time:.3f}s")
    project.info(f"Average time per image: {avg_time_per_image:.3f}s")
    project.info(f"Total images generated: {test_num * batch_size}")
    project.info(f"Images per second: {test_num * batch_size / total_time:.2f}")
    project.info(f"{'='*60}")

    # Evaluate images if requested
    evaluation_results = None
    if evaluate and all_images:
        evaluation_results = evaluate_images(all_images, sampler_type)

    return {
        'total_time': total_time,
        'avg_time_per_batch': avg_time,
        'avg_time_per_image': avg_time_per_image,
        'total_images': test_num * batch_size,
        'images_per_second': test_num * batch_size / total_time,
        'generation_times': generation_times,
        'evaluation_results': evaluation_results
    }

def save_generation_log(save_dir, sampler_type, generation_stats, args):
    """Save generation log to JSON file"""

    log_data = {
        'timestamp': datetime.now().isoformat(),
        'sampler_type': sampler_type,
        'sampler_description': get_sampler_description(sampler_type),
        'parameters': {
            'test_num': args.test_num,
            'start_index': args.start_index,
            'batch_size': args.batch_size,
            'num_inference_steps': args.num_inference_steps,
            'lamb': args.lamb,
            'kappa': args.kappa,
            'dtype': args.dtype,
            'device': args.device
        },
        'generation_stats': generation_stats
    }

    log_filename = f"generation_log_{sampler_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path = os.path.join(save_dir, log_filename)

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    project.info(f"  ✓ Generation log saved to: {log_path}")

def count_generated_images(save_dir):
    """Count the number of generated images in the save directory"""
    try:
        png_files = glob.glob(os.path.join(save_dir, "*.png"))
        return len(png_files)
    except Exception as e:
        project.warning(f"Could not count images in {save_dir}: {e}")
        return 0

def run_single_experiment(args, experiment_num, total_experiments, sampler_type, num_inference_steps):
    """Run a single experiment with enhanced logging"""

    project.info("")
    project.info("="*50)
    project.info(f"实验 {experiment_num}/{total_experiments}")
    project.info(f"Sampler: {sampler_type}")
    project.info(f"Steps: {num_inference_steps}")
    project.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    project.info("="*50)

    # 记录实验开始时间
    exp_start_time = time.time()

    try:
        # Convert dtype string to torch dtype
        dtype_map = {
            'fp32': torch.float32,
            'fp64': torch.float64,
            'fp16': torch.float16,
            'bf16': torch.bfloat16
        }
        dtype = dtype_map[args.dtype]

        # Setup paths
        model_id = celeba_model_id
        save_dir = os.path.join(project.output_dir, args.save_dir, "steps"+'_'+str(num_inference_steps), sampler_type)
        os.makedirs(save_dir, exist_ok=True)

        project.info(f"🚀 CelebA-HQ Unified Sampling Script")
        project.info("="*60)
        project.info(f"Model: {model_id}")
        project.info(f"Device: {args.device}")
        project.info(f"Data type: {args.dtype}")
        project.info(f"Sampler: {sampler_type}")
        project.info(f"Output: {save_dir}")
        project.info("="*60)

        # Load pipeline
        project.info("\n📦 Loading model...")
        pipe = LDMPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=False)
        pipe.unet.to(args.device)
        pipe.vqvae.to(args.device)
        project.success("Model loaded successfully")

        # Setup scheduler
        project.info(f"\n⚙️ Setting up scheduler...")
        setup_scheduler(pipe, sampler_type, args.lamb, args.kappa)

        # Generate images
        generation_stats = generate_images(
            pipe, args.batch_size, num_inference_steps,
            args.test_num, args.start_index, save_dir, sampler_type, args.evaluate
        )

        # Save generation log if requested
        if args.save_log:
            project.info(f"\n📝 Saving generation log...")
            save_generation_log(save_dir, sampler_type, generation_stats, args)

        # 计算实验耗时
        exp_end_time = time.time()
        exp_duration = exp_end_time - exp_start_time

        # 统计生成的图片数量
        image_count = count_generated_images(save_dir)

        project.success(f"实验完成! 耗时: {exp_duration:.1f}秒")
        project.info(f"生成图片数量: {image_count}")
        project.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            'success': True,
            'duration': exp_duration,
            'image_count': image_count,
            'generation_stats': generation_stats,
            'save_dir': save_dir
        }

    except Exception as e:
        exp_end_time = time.time()
        exp_duration = exp_end_time - exp_start_time

        project.error(f"实验失败! 耗时: {exp_duration:.1f}秒")
        project.error(f"错误信息: {str(e)}")
        project.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            'success': False,
            'duration': exp_duration,
            'error': str(e),
            'sampler_type': sampler_type,
            'num_inference_steps': num_inference_steps
        }

def generate_experiment_summary(experiment_results, start_time, end_time, args):
    """Generate comprehensive experiment summary report"""

    summary_file = os.path.join(project.output_dir, args.save_dir, "experiment_summary.txt")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    successful_experiments = [r for r in experiment_results if r['success']]
    failed_experiments = [r for r in experiment_results if not r['success']]

    total_images = sum(r.get('image_count', 0) for r in successful_experiments)
    total_duration = sum(r.get('duration', 0) for r in experiment_results)

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("CelebA-HQ 实验总结报告\n")
        f.write("="*50 + "\n\n")
        f.write(f"运行时间: {start_time} 到 {end_time}\n")
        f.write(f"总实验数: {len(experiment_results)}\n")
        f.write(f"成功实验: {len(successful_experiments)}\n")
        f.write(f"失败实验: {len(failed_experiments)}\n")
        f.write(f"总生成图片: {total_images}\n")
        f.write(f"总耗时: {total_duration:.1f}秒\n\n")

        f.write("参数配置:\n")
        f.write(f"- Batch Size: {args.batch_size}\n")
        f.write(f"- Test Num: {args.test_num}\n")
        f.write(f"- Device: {args.device}\n")
        f.write(f"- Data Type: {args.dtype}\n")
        f.write(f"- Lambda: {args.lamb}\n")
        f.write(f"- Kappa: {args.kappa}\n\n")

        f.write("各实验详情:\n")
        for result in experiment_results:
            if result['success']:
                f.write(f"- {result.get('sampler_type', 'unknown')} ({result.get('num_inference_steps', 'unknown')} steps): {result.get('image_count', 0)} 张图片, 耗时 {result.get('duration', 0):.1f}秒\n")
            else:
                f.write(f"- {result.get('sampler_type', 'unknown')} ({result.get('num_inference_steps', 'unknown')} steps): 失败 - {result.get('error', 'Unknown error')}\n")

    project.info(f"\n📊 实验总结报告已保存到: {summary_file}")

    # 保存失败实验日志
    if failed_experiments:
        failed_log_file = os.path.join(project.output_dir, args.save_dir, "failed_experiments.log")
        with open(failed_log_file, 'w', encoding='utf-8') as f:
            for result in failed_experiments:
                f.write(f"{result.get('sampler_type', 'unknown')},{result.get('num_inference_steps', 'unknown')},{result.get('error', 'Unknown error')}\n")
        project.warning(f"有 {len(failed_experiments)} 个实验失败，详情请查看: {failed_log_file}")
    else:
        project.success("所有实验都成功完成!")

def compare_all_samplers(args):
    """Compare all available samplers and generate comparison table"""

    samplers = ['ddim', 'pndm', 'dpm', 'dpm++', 'unipc', 'dpm_lm', 'hessian_free']
    results_dict = {}

    project.info("🔄 Running comprehensive comparison of all samplers...")
    project.info("="*80)

    for sampler in samplers:
        project.info(f"\n🔍 Testing {sampler}...")

        # Update args for this sampler
        args.sampler_type = sampler
        args.test_num = 5  # Reduced for comparison
        args.evaluate = True

        try:
            # Run generation and evaluation
            generation_stats = run_single_sampler(args)
            if generation_stats and generation_stats.get('evaluation_results'):
                results_dict[sampler] = generation_stats['evaluation_results']
        except Exception as e:
            project.error(f"Error with {sampler}: {e}")
            continue

    # Generate comparison table
    if results_dict:
        table_data = generate_comparison_table(results_dict)
        format_table_with_ranking(table_data)

        # Save results
        if args.save_results:
            results_file = os.path.join(project.output_dir, 'celeba_comparison_results.json')
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            project.info(f"\n📊 Results saved to: {results_file}")

    return results_dict

def run_single_sampler(args):
    """Run a single sampler configuration"""

    # Convert dtype string to torch dtype
    dtype_map = {
        'fp32': torch.float32,
        'fp64': torch.float64,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]

        # Setup paths
    model_id = celeba_model_id
    save_dir = os.path.join(project.output_dir, args.save_dir,  "steps"+'_'+str(args.num_inference_steps), args.sampler_type)
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Load pipeline
        pipe = LDMPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=False)
        pipe.unet.to(args.device)
        pipe.vqvae.to(args.device)

        # Setup scheduler
        setup_scheduler(pipe, args.sampler_type, args.lamb, args.kappa)

        # Generate images
        generation_stats = generate_images(
            pipe, args.batch_size, args.num_inference_steps,
            args.test_num, args.start_index, save_dir, args.sampler_type, args.evaluate
        )

        # Save generation log if requested
        if args.save_log:
            save_generation_log(save_dir, args.sampler_type, generation_stats, args)

        return generation_stats

    except Exception as e:
        project.error(f"Error during generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None

def main(args):
    """Main function for single experiment"""
    project.info(args)

    # Convert dtype string to torch dtype
    dtype_map = {
        'fp32': torch.float32,
        'fp64': torch.float64,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]

    # Setup paths
    model_id = celeba_model_id
    save_dir = os.path.join(project.output_dir, args.save_dir,  "steps"+'_'+str(args.num_inference_steps), args.sampler_type)
    os.makedirs(save_dir, exist_ok=True)

    project.info("🚀 CelebA-HQ Unified Sampling Script")
    project.info("="*60)
    project.info(f"Model: {model_id}")
    project.info(f"Device: {args.device}")
    project.info(f"Data type: {args.dtype}")
    project.info(f"Sampler: {args.sampler_type}")
    project.info(f"Output: {save_dir}")
    project.info("="*60)

    # Load pipeline
    project.info("\n📦 Loading model...")
    pipe = LDMPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=False)
    pipe.unet.to(args.device)
    pipe.vqvae.to(args.device)
    project.success("Model loaded successfully")

    # Setup scheduler
    project.info(f"\n⚙️ Setting up scheduler...")
    setup_scheduler(pipe, args.sampler_type, args.lamb, args.kappa)

    # Generate images
    generation_stats = generate_images(
        pipe, args.batch_size, args.num_inference_steps,
        args.test_num, args.start_index, save_dir, args.sampler_type, args.evaluate
    )

    # Save generation log if requested
    if args.save_log:
        project.info(f"\n📝 Saving generation log...")
        save_generation_log(save_dir, args.sampler_type, generation_stats, args)

    project.success("Generation completed successfully!")
    project.info(f"   Generated {generation_stats['total_images']} images")
    project.info(f"   Total time: {generation_stats['total_time']:.2f}s")
    project.info(f"   Average time per image: {generation_stats['avg_time_per_image']:.3f}s")

    if generation_stats.get('evaluation_results'):
        project.info(f"\n📊 Evaluation Results:")
        eval_results = generation_stats['evaluation_results']
        project.info(f"   ColorS: {eval_results['ColorS']:.3f}")
        project.info(f"   FS: {eval_results['FS']:.3f}")
        project.info(f"   DFIQA: {eval_results['DFIQA']:.3f}")
        project.info(f"   PicS: {eval_results['PicS']:.3f}")
        project.info(f"   EAT: {eval_results['EAT']:.3f}")
        project.info(f"   Laion: {eval_results['Laion']:.3f}")

if __name__ == '__main__':
    # 设置日志系统
    logger = project.setup_logging(name='celeba', level=project.logging.INFO)

    args = parse_args()

    # 检查是否运行批量实验
    if args.run_batch:
        # 批量实验模式
        SAMPLER_TYPES = ["pndm", "ddim", "dpm++", "dpm", "unipc", "hessian_free"]
        # INFERENCE_STEPS = [5, 7, 9, 12, 15, 20, 50]
        # INFERENCE_STEPS = [5, 6]

        project.log_experiment_start("CelebA-HQ 批量实验", {
            'samplers': SAMPLER_TYPES,
            'steps': args.num_inference_steps,
            'batch_size': args.batch_size,
            'test_num': args.test_num
        })

        total_experiments = len(SAMPLER_TYPES)
        experiment_results = []
        start_time = datetime.now()

        experiment_num = 0

        for j in range(len(SAMPLER_TYPES)):
            experiment_num += 1
            num_inference_steps = args.num_inference_steps
            sampler_type = SAMPLER_TYPES[j]

            # 更新args
            args.num_inference_steps = num_inference_steps
            args.sampler_type = sampler_type

            project.progress(experiment_num, total_experiments, f"Running {sampler_type} with {num_inference_steps} steps")

            result = run_single_experiment(args, experiment_num, total_experiments, sampler_type, num_inference_steps)
            experiment_results.append(result)

        end_time = datetime.now()

        # 生成实验总结报告
        project.info("\n生成实验总结报告...")
        generate_experiment_summary(experiment_results, start_time, end_time, args)

        project.log_experiment_end("CelebA-HQ 批量实验",
                                 duration=(end_time - start_time).total_seconds(),
                                 results={'total_experiments': total_experiments, 'successful': len([r for r in experiment_results if r['success']])})

        project.info(f"\n实验完成! 结果保存在: {os.path.join(project.output_dir, args.save_dir)}")
        project.info(f"日志文件: {logger.get_log_file()}")
    else:
        # 单个实验模式
        if args.compare_all:
            # Compare all samplers
            results_dict = compare_all_samplers(args)
            project.success("Comparison completed!")
        else:
            # Run single sampler
            generation_stats = run_single_sampler(args)

            if generation_stats:
                project.success("Generation completed successfully!")
                project.info(f"   Generated {generation_stats['total_images']} images")
                project.info(f"   Total time: {generation_stats['total_time']:.2f}s")
                project.info(f"   Average time per image: {generation_stats['avg_time_per_image']:.3f}s")

                if generation_stats.get('evaluation_results'):
                    project.info(f"\n📊 Evaluation Results:")
                    eval_results = generation_stats['evaluation_results']
                    project.info(f"   ColorS: {eval_results['ColorS']:.3f}")
                    project.info(f"   FS: {eval_results['FS']:.3f}")
                    project.info(f"   DFIQA: {eval_results['DFIQA']:.3f}")
                    project.info(f"   PicS: {eval_results['PicS']:.3f}")
                    project.info(f"   EAT: {eval_results['EAT']:.3f}")
                    project.info(f"   Laion: {eval_results['Laion']:.3f}")

    # 关闭日志器
    logger.close()
