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
import pandas as pd

sys.path.append(os.getcwd())
from diffusers import LDMPipeline, DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler
from scheduler.scheduling_dpmsolver_multistep_hcg import DPMSolverMultistepHCGScheduler

# Import profiling utilities
try:
    from utils.profiling import get_profiler, profile
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
from evaluations.celeba_eva import evaluate_images
import project as project

celeba_model_path = "/home/liying/Documents/ldm-celebahq-256/"
# celeba_model_path = "/research/cbim/vast/cj574/data/diffusion/celeba_hq_256/ldm-celebahq-256"


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="CelebA-HQ sampling script with enhanced features")

    # Basic parameters
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=50, choices=[5, 10, 20, 40, 50, 70, 100, 200, 400, 600, 1000])

    parser.add_argument('--scaling_factor', type=float, default=0.18215)
    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--seed', type=int, default=102)

    # Sampler selection
    parser.add_argument('--sampler_type', type=str, default='dpm_hcg',
                        choices=['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'dpm_hcg'])
    parser.add_argument('--use_generator', action='store_true', default=True)

    # Output configuration
    parser.add_argument('--save_dir', type=str, default='celebahq_256')
    parser.add_argument('--model_path', type=str, default=celeba_model_path)
    parser.add_argument('--model_type', type=str, default="ldm")

    # LML parameters
    parser.add_argument('--lamb', type=float, default=0.004)
    parser.add_argument('--kappa', type=float, default=1.0e-8)

    # Technical parameters
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp64', 'fp16', 'bf16'])
    parser.add_argument('--device', type=str, default='cuda')

    # HCG (Hessian-Conjugate Gradient) parameters
    parser.add_argument('--kappa_target', type=float, default=100.0, help='Target condition number for adaptive damping')
    parser.add_argument('--lanczos_k', type=int, default=5, help='Number of Lanczos iterations for eigenvalue estimation')
    parser.add_argument('--cg_max_iter', type=int, default=5, help='Maximum CG iterations')
    parser.add_argument('--cg_tol', type=float, default=1e-3, help='CG tolerance (default: 1e-3, tighter than 1e-2 for better convergence)')
    # HCG (Hessian-Conjugate Gradient) parameters for spectral radius scaling
    parser.add_argument('--use_spectral_scaling', action='store_true', default=True, help='Use spectral radius scaling c_t = beta_t + lambda_t (default: True, REQUIRED for HCG to work)')
    parser.add_argument('--spectral_scaling', type=float, default=1.0, help='Spectral radius scaling factor when use_spectral_scaling=False (default: 1.0, only used if disabled)')

    # HCG (Hessian-Conjugate Gradient) parameters for adaptive damping
    parser.add_argument('--use_adaptive_lambda', action='store_true', default=False, help='Use adaptive lambda_t (default: False, optimized). If False, use fixed lambda_t = lambda_base.')
    parser.add_argument('--lambda_base', type=float, default=0.004, help='Base lambda_t (default: 0.3, optimized). If use_adaptive_lambda=False, use fixed lambda_t = lambda_base.')
    parser.add_argument('--lambda_scale', type=float, default=0.3, help='Scaling factor for adaptive lambda_t (default: 0.3, optimized). If use_adaptive_lambda=True, use adaptive lambda_t = lambda_scale * lambda_t_base.')

    parser.add_argument('--log_lambda_stats', action='store_true', default=True, help='Log lambda_t statistics to understand its range during sampling (default: True).')
    parser.add_argument('--enable_eigenvalue_cache', action='store_true', default=True, help='Enable eigenvalue caching for faster computation (default: True).')
    parser.add_argument('--eigenvalue_cache_interval', type=int, default=5, help='Re-estimate eigenvalues every N steps when caching is enabled (default: 5). If use_adaptive_lambda=True, re-estimate eigenvalues every N steps when caching is enabled.')

    # Evaluation options
    parser.add_argument('--evaluate', action='store_true', default=False, help='Run evaluation metrics')
    parser.add_argument('--generate_grid', action='store_true', default=False, help='Generate comparison grid from existing images')
    parser.add_argument('--grid_title', type=str, default="CelebA-HQ Generation Comparison", help='Title of comparison grid')
    parser.add_argument('--grid_test_num', type=int, default=6, help='Number of images to test in grid')
    parser.add_argument('--grid_test_index', type=list, default=[0, 1, 2, 3, 4, 5], help='Index of images to test in grid')
    parser.add_argument('--grid_samplers', default=['ddim', 'pndm', 'dpm++', 'dpm', 'unipc', 'dpm_hcg'],
                        help='List of samplers to test in batch mode')

    # Profiling options
    parser.add_argument('--enable_profiling', action='store_true', default=True, help='Enable function-level profiling (similar to MATLAB profiler)')
    parser.add_argument('--profile_report_file', type=str, default=None, help='Path to save profiling report (JSON format)')

    # Logging options
    parser.add_argument('--log_dir', type=str, default=None, help='Directory to save log files (default: output/logs)')

    # Batch processing options
    parser.add_argument('--run_batch', action='store_true', default=False, help='Run batch experiments with multiple samplers and steps')
    parser.add_argument('--run_batch_samplers', default=['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'dpm_hcg'], help='List of samplers to test in batch mode')
    parser.add_argument('--run_batch_steps', type=int, default=[20, 50, 200], help='List of inference steps to test in batch mode')

    # Additional options
    parser.add_argument('--save_log', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true')

    # Performance analysis options
    parser.add_argument('--analyze_performance', action='store_true', default=False, help='Run performance analysis after experiments')
    parser.add_argument('--performance_output_dir', type=str, default='output/celeba', help='Directory for performance analysis output')
    parser.add_argument('--performance_only', action='store_true', default=False, help='Only run performance analysis on existing results')

    args = parser.parse_args()

    return args

def setup_scheduler(pipe, sampler_type, lamb=0.0008, kappa=1e-8):
    """Setup the appropriate scheduler based on sampler type"""
    # 获取原始配置并过滤掉运行时状态属性（避免警告）
    original_config = pipe.scheduler.config
    config_dict = {
        k: v for k, v in original_config.items()
        if k not in ['timestep_values', 'timesteps']  # 移除这些运行时状态属性
    }

    if sampler_type == 'pndm':
        pipe.scheduler = PNDMScheduler.from_config(config_dict)
        project.info(f"  Using PNDM scheduler")

    elif sampler_type == 'ddim':
        pipe.scheduler = DDIMScheduler.from_config(config_dict)
        pipe.scheduler.config.eta = 0.0  # 设置eta=0.0，与celeba_test.py保持一致
        project.info(f"  Using DDIM scheduler (eta=0.0)")

    elif sampler_type == 'ddim_lm':
        pipe.scheduler = DDIMLMScheduler.from_config(config_dict)
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = True
        pipe.scheduler.kappa = kappa
        project.info(f"  Using DDIM with LML correction (λ={lamb}, κ={kappa})")

    elif sampler_type == 'dpm++':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(config_dict)
        pipe.scheduler.config.algorithm_type = "dpmsolver++"
        pipe.scheduler.config.solver_order = 3
        project.info(f"  Using DPM-Solver++ scheduler")

    elif sampler_type == 'dpm_lm':
        pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(config_dict)
        pipe.scheduler.config.algorithm_type = "dpmsolver"
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = True
        pipe.scheduler.kappa = kappa
        project.info(f"  Using DPM-Solver with LML correction (λ={lamb}, κ={kappa})")

    elif sampler_type == 'dpm':
        pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(config_dict)
        pipe.scheduler.config.algorithm_type = "dpmsolver"
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.lm = False
        project.info(f"  Using DPM-Solver scheduler")

    elif sampler_type == 'unipc':
        pipe.scheduler = UniPCMultistepScheduler.from_config(config_dict)
        project.info(f"  Using UniPC scheduler")

    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

def setup_scheduler_hcg(pipe, kappa_target=20.0, lanczos_k=5, cg_max_iter=5,
                    cg_tol=1e-3, use_spectral_scaling=True, spectral_scaling=1.0,
                    use_adaptive_lambda=True, lambda_base=0.004, lambda_scale=0.3,
                    log_lambda_stats=True, enable_eigenvalue_cache=True,
                    eigenvalue_cache_interval=5):
    """Setup the HCG scheduler"""
    # 获取原始配置并过滤掉不需要的属性（避免警告）
    original_config = pipe.scheduler.config
    # 创建新的配置字典，只包含 DPMSolverMultistepHCGScheduler 需要的字段
    config_dict = {
        k: v for k, v in original_config.items()
        if k not in ['timestep_values', 'timesteps']  # 移除这些运行时状态属性
    }

    # 使用新的 DPMSolverMultistepHCGScheduler
    pipe.scheduler = DPMSolverMultistepHCGScheduler.from_config(config_dict)
    pipe.scheduler.config.algorithm_type = "dpmsolver++"
    pipe.scheduler.config.solver_order = 3

    # 设置模型用于Hessian计算（必需）
    pipe.scheduler.set_model(pipe.unet)

    # 设置HCG参数
    pipe.scheduler.use_hcg = True
    pipe.scheduler.kappa_target = kappa_target
    pipe.scheduler.lanczos_k = lanczos_k
    pipe.scheduler.cg_max_iter = cg_max_iter
    pipe.scheduler.cg_tol = cg_tol
    pipe.scheduler.use_spectral_scaling = use_spectral_scaling
    pipe.scheduler.spectral_scaling = spectral_scaling
    pipe.scheduler.use_adaptive_lambda = use_adaptive_lambda
    pipe.scheduler.lambda_base = lambda_base
    pipe.scheduler.lambda_scale = lambda_scale
    pipe.scheduler.log_lambda_stats = log_lambda_stats
    pipe.scheduler.enable_eigenvalue_cache = enable_eigenvalue_cache
    pipe.scheduler.eigenvalue_cache_interval = eigenvalue_cache_interval

    project.info(f"  Using DPM-Solver++ with HCG (Hessian-Conjugate Gradient) correction")
    project.info(f"    kappa_target={kappa_target}, lanczos_k={lanczos_k}, cg_max_iter={cg_max_iter}")
    project.info(f"    cg_tol={cg_tol}, use_spectral_scaling={use_spectral_scaling}, spectral_scaling={spectral_scaling}")
    project.info(f"    use_adaptive_lambda={use_adaptive_lambda}, lambda_base={lambda_base}, lambda_scale={lambda_scale:.4f}, log_lambda_stats={log_lambda_stats}")
    project.info(f"    enable_eigenvalue_cache={enable_eigenvalue_cache}, cache_interval={eigenvalue_cache_interval}")

def process_image(image):
    """
    Process VAE decoded tensor to PIL Image

    Args:
        image: torch.Tensor of shape (B, C, H, W) with values in [-1, 1] range

    Returns:
        PIL.Image ready to save
    """
    # image is a tensor of shape (B, C, H, W) with values in [-1, 1]
    # Convert to (B, H, W, C) format
    image_processed = image.permute(0, 2, 3, 1)
    # Convert from [-1, 1] to [0, 255]
    image_processed = (image_processed + 1.0) * 127.5
    # Clamp and convert to uint8
    image_processed = image_processed.clamp(0, 255).cpu().numpy().astype(np.uint8)
    # Get first image from batch
    image_pil = Image.fromarray(image_processed[0])

    return image_pil

def generate_images(results_save_dir, args, pipe):
    """Generate images using the specified pipeline"""

    # Enable profiling if requested
    if PROFILING_AVAILABLE and args.enable_profiling:
        profiler = get_profiler()
        profiler.enabled = True
        profiler.clear()
        project.info("📊 Profiling enabled - will track function execution times")

    total_time = 0
    generation_times = []
    all_images = []

    project.info(f"\n{'='*60}")
    project.info(f"Starting image generation")
    project.info(f"{'='*60}")
    project.info(f"Sampler: {args.sampler_type} - {project.get_sampler_description(args.sampler_type)}")
    project.info(f"Batch size: {args.batch_size}")
    project.info(f"Inference steps: {args.num_inference_steps}")
    project.info(f"Total images: {args.test_num}")
    project.info(f"Save directory: {results_save_dir}")
    project.info(f"{'='*60}")

    for seed in range(args.seed, args.seed + args.test_num):
        project.info(f"\nGenerating batch {seed - args.seed + 1}/{args.test_num} (seed={seed})")
        batch_start_time = time.time()
        torch.manual_seed(seed)

        # Generate images
        # 确保方法有正确的模型设置
        # if hasattr(pipe.scheduler, 'set_model') and pipe.scheduler.model is None:
        #     pipe.scheduler.set_model(pipe.unet)

        # HCG method requires gradient computation for Hessian-vector products
        # So we can't use torch.no_grad() for HCG sampler
        if args.sampler_type == 'dpm_hcg' and pipe.scheduler.model is not None:
            # For HCG, we need gradients enabled for Hessian computation
            # But we still want to disable gradients for final output
            images = pipe(batch_size=args.batch_size, num_inference_steps=args.num_inference_steps).images
            # Detach images to free memory
            if isinstance(images, torch.Tensor):
                images = images.detach().cpu()
        else:
            # For other samplers, use no_grad for efficiency
            with torch.no_grad():
                images = pipe(batch_size=args.batch_size, num_inference_steps=args.num_inference_steps).images

        # Save images
        for i, image in enumerate(images):
            filename = f"celeba_{args.sampler_type}_inference{args.num_inference_steps}_seed{seed}_{i}.png"
            project.info(f"  ✓ Saved to: {filename}")
            filepath = os.path.join(results_save_dir, filename)
            image.save(filepath)
            all_images.append(image)

        batch_time = time.time() - batch_start_time
        generation_times.append(batch_time)
        total_time += batch_time

        project.info(f"  ✓ Generated {len(images)} images in {batch_time:.3f}s")
        project.info(f"  ✓ Saved to: {results_save_dir}")

    # Print summary
    avg_time = total_time / args.test_num
    avg_time_per_image = avg_time / args.batch_size

    project.info(f"\n{'='*60}")
    project.info(f"GENERATION SUMMARY")
    project.info(f"{'='*60}")
    project.info(f"Total time: {total_time:.2f}s")
    project.info(f"Average time per batch: {avg_time:.3f}s")
    project.info(f"Average time per image: {avg_time_per_image:.3f}s")
    project.info(f"Total images generated: {args.test_num * args.batch_size}")
    project.info(f"Images per second: {args.test_num * args.batch_size / total_time:.2f}")
    project.info(f"{'='*60}")

    # Print HCG statistics if using HCG sampler
    if args.sampler_type == 'dpm_hcg':
        if hasattr(pipe.scheduler, 'get_hcg_statistics'):
            hcg_stats = pipe.scheduler.get_hcg_statistics()
            if hcg_stats and hcg_stats.get('intermediate_vars'):
                vars_dict = hcg_stats['intermediate_vars']
                timesteps_history = vars_dict.get('timesteps_history', [])

                if timesteps_history and len(timesteps_history) > 0:
                    project.info("\n" + "="*60)
                    project.info("HCG INTERMEDIATE STATISTICS SUMMARY")
                    project.info("="*60)

                    # Extract key statistics
                    kappa_reg = vars_dict.get('kappa_regularized_history', [])
                    kappa_orig = vars_dict.get('kappa_original_history', [])
                    kappa_target = vars_dict.get('kappa_target_history', [])
                    lambda_t_list = vars_dict.get('lambda_t_history', [])
                    c_t_list = vars_dict.get('c_t_history', [])
                    cg_iter = vars_dict.get('cg_iterations_history', [])
                    cg_residual = vars_dict.get('cg_final_residual_history', [])
                    k_pred = vars_dict.get('theoretical_min_k_history', [])
                    alpha_t_list = vars_dict.get('alpha_t_history', [])
                    beta_t_list = vars_dict.get('beta_t_history', [])

                    if kappa_reg:
                        kappa_target_val = kappa_target[0] if kappa_target and len(kappa_target) > 0 else None
                        project.info(f"Condition Numbers (κ):")
                        if kappa_target_val is not None:
                            project.info(f"  Target κ*: {kappa_target_val:.2f}")
                        project.info(f"  Original κ(H_sym) - avg: {np.mean(kappa_orig):.2f}, min: {np.min(kappa_orig):.2f}, max: {np.max(kappa_orig):.2f}")
                        project.info(f"  Regularized κ(A_t) - avg: {np.mean(kappa_reg):.2f}, min: {np.min(kappa_reg):.2f}, max: {np.max(kappa_reg):.2f}")
                        if kappa_target_val is not None:
                            violations = sum(1 for k in kappa_reg if k > kappa_target_val)
                            project.info(f"  κ(A_t) ≤ κ* violations: {violations}/{len(kappa_reg)}")

                    if lambda_t_list:
                        project.info(f"\n")
                        project.info(f"Adaptive Damping (λ_t):")
                        project.info(f"  Mean: {np.mean(lambda_t_list):.6f}, Min: {np.min(lambda_t_list):.6f}, Max: {np.max(lambda_t_list):.6f}")
                        project.info(f"  Zero damping steps: {sum(1 for l in lambda_t_list if abs(l) < 1e-8)}/{len(lambda_t_list)}")

                    if c_t_list:
                        project.info(f"\n")
                        project.info(f"Spectral Scaling (c_t):")
                        project.info(f"  Mean: {np.mean(c_t_list):.6f}, Min: {np.min(c_t_list):.6f}, Max: {np.max(c_t_list):.6f}")

                    if cg_iter and k_pred:
                        project.info(f"\n")
                        project.info(f"CG Convergence:")
                        project.info(f"  Actual iterations (k_obs) - avg: {np.mean(cg_iter):.2f}, min: {int(np.min(cg_iter))}, max: {int(np.max(cg_iter))}")
                        project.info(f"  Theoretical lower bound (k_pred) - avg: {np.mean(k_pred):.2f}, min: {np.min(k_pred):.2f}, max: {np.max(k_pred):.2f}")
                        cg_tol_val = getattr(args, 'cg_tol', 1e-2)
                        convergence_rate = sum(1 for r in cg_residual if r < cg_tol_val)
                        project.info(f"  Convergence rate: {convergence_rate}/{len(cg_residual)} (residual < {cg_tol_val})")

                    if alpha_t_list and beta_t_list:
                        project.info(f"\n")
                        project.info(f"Eigenvalue Estimates:")
                        project.info(f"  α_t (max) - avg: {np.mean(alpha_t_list):.6f}, min: {np.min(alpha_t_list):.6f}, max: {np.max(alpha_t_list):.6f}")
                        project.info(f"  β_t (min) - avg: {np.mean(beta_t_list):.6f}, min: {np.min(beta_t_list):.6f}, max: {np.max(beta_t_list):.6f}")

                    project.info(f"\n")
                    project.info(f"Total HVP calls: {vars_dict.get('hvp_call_count', 0)}")
                    project.info(f"Eigenvalue estimations: {vars_dict.get('eigenvalue_estimates_count', 0)}")
                    project.info(f"Total timesteps processed: {len(timesteps_history)}")
                    project.info("="*60 + "\n")
                else:
                    project.warning("HCG statistics collection is enabled but no data was collected. "
                                  f"collect_stats={getattr(pipe.scheduler, 'collect_stats', 'N/A')}, "
                                  f"timesteps_history length={len(timesteps_history)}")
            else:
                project.warning("HCG sampler detected but hcg_stats is empty or missing intermediate_vars")
        else:
            project.warning(f"HCG sampler detected but scheduler does not have get_hcg_statistics method")

    # Print profiling report if enabled
    if PROFILING_AVAILABLE and args.enable_profiling:
        profiler = get_profiler()
        if profiler and profiler.enabled:
            project.info("\n" + "="*60)
            project.info("PROFILING REPORT")
            project.info("="*60)
            # Pass logger - it will output once to both console and file
            profiler.print_report(sort_by='total_time', top_n=20, logger=project.logger)

            # Save profiling report if requested
            if args.profile_report_file:
                profiler.save_report(args.profile_report_file)
            else:
                # Auto-save to results directory
                profile_file = os.path.join(results_save_dir, f"profiling_report_{args.sampler_type}.json")
                profiler.save_report(profile_file)

    # Evaluate images if requested
    evaluation_results = None
    if args.evaluate and all_images:
        evaluation_results = evaluate_images(all_images, args.sampler_type)

    return {
        'total_time': total_time,
        'avg_time_per_batch': avg_time,
        'avg_time_per_image': avg_time_per_image,
        'total_images': args.test_num * args.batch_size,
        'images_per_second': args.test_num * args.batch_size / total_time,
        'generation_times': generation_times,
        'evaluation_results': evaluation_results
    }

def run_single_experiment(results_save_dir, args, experiment_num, total_experiments):
    """Run a single experiment with enhanced logging"""

    project.info("")
    project.info("="*50)
    project.info(f"实验 {experiment_num}/{total_experiments}")
    project.info(f"Sampler: {args.sampler_type}")
    project.info(f"Steps: {args.num_inference_steps}")
    project.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    project.info("="*50)

    # 记录实验开始时间
    exp_start_time = time.time()

    # Convert dtype string to torch dtype
    dtype_map = {
        'fp32': torch.float32,
        'fp64': torch.float64,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]

    # Setup paths
    model_path = args.model_path
    results_save_dir = os.path.join(results_save_dir, "steps"+'_'+str(args.num_inference_steps), args.sampler_type)
    os.makedirs(results_save_dir, exist_ok=True)

    project.info(f"🚀 CelebA-HQ Unified Sampling Script")
    project.info("="*60)
    project.info(f"Model: {model_path}")
    project.info(f"Device: {args.device}")
    project.info(f"Data type: {args.dtype}")
    project.info(f"Sampler: {args.sampler_type}")
    project.info(f"Output: {results_save_dir}")
    project.info("="*60)

    # Load pipeline
    project.info("\n📦 Loading model...")
    pipe = LDMPipeline.from_pretrained(model_path, torch_dtype=dtype, use_safetensors=False)
    pipe.unet.to(args.device)
    pipe.vqvae.to(args.device)
    pipe.vqvae.config.scaling_factor = 1.0 # args.scaling_factor

    project.success("Model loaded successfully")

    # Setup scheduler
    project.info(f"\n⚙️ Setting up scheduler...")
    if args.sampler_type == 'dpm_hcg':
        setup_scheduler_hcg(
            pipe, kappa_target=args.kappa_target,
            lanczos_k=args.lanczos_k,
            cg_max_iter=args.cg_max_iter,
            cg_tol=args.cg_tol,
            use_spectral_scaling=args.use_spectral_scaling,
            spectral_scaling=args.spectral_scaling,
            use_adaptive_lambda=args.use_adaptive_lambda,
            lambda_base=args.lambda_base,
            lambda_scale=args.lambda_scale,
            log_lambda_stats=args.log_lambda_stats,
            enable_eigenvalue_cache=args.enable_eigenvalue_cache,
            eigenvalue_cache_interval=args.eigenvalue_cache_interval
        )
    else:
        setup_scheduler(pipe, args.sampler_type, args.lamb, args.kappa)

    # Generate images
    generation_stats = generate_images(
        results_save_dir, args, pipe
    )

    # Save generation log if requested
    if args.save_log:
        project.info(f"\n📝 Saving generation log...")
        project.save_generation_log(results_save_dir, args, generation_stats)

    # 计算实验耗时
    exp_end_time = time.time()
    exp_duration = exp_end_time - exp_start_time

    # 统计生成的图片数量
    # image_count = project.count_generated_images(results_save_dir)

    project.success(f"实验完成! 耗时: {exp_duration:.1f}秒")
    # project.info(f"生成图片数量: {image_count}")
    project.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return {
        'success': True,
        'duration': exp_duration,
        # 'image_count': image_count,
        'generation_stats': generation_stats,
        'results_save_dir': results_save_dir
    }


if __name__ == '__main__':
    # 先解析参数（但不解析 log_dir，因为日志需要在参数解析前设置）
    # 我们需要先设置一个默认的日志系统，然后再重新设置
    args = parse_args()

    # 设置日志系统（使用解析后的 log_dir）
    logger = project.setup_logging(name='celeba', log_dir=args.log_dir, level=project.logging.INFO)
    results_save_dir = os.path.join(project.output_dir, args.save_dir + '_' + args.model_type)

    # 单个实验模式（保持原有逻辑）
    SAMPLER_TYPES = [args.sampler_type]
    INFERENCE_STEPS = [args.num_inference_steps]

    # 检查是否运行批量实验
    if hasattr(args, 'run_batch') and args.run_batch:
        # 批量实验模式
        SAMPLER_TYPES = args.run_batch_samplers  #['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'dpm_hcg']
        INFERENCE_STEPS = args.run_batch_steps  # [5, 10, 20, 30]

    total_experiments = len(SAMPLER_TYPES) * len(INFERENCE_STEPS)
    experiment_results = []
    start_time = datetime.now()
    experiment_num = 0

    project.log_experiment_start("CelebA-HQ 批量实验", {
        'samplers': SAMPLER_TYPES,
        'steps': args.num_inference_steps,
        'batch_size': args.batch_size,
        'test_num': args.test_num
    })

    for i in range(len(INFERENCE_STEPS)):
        for j in range(len(SAMPLER_TYPES)):
            experiment_num += 1
            num_inference_steps = INFERENCE_STEPS[i]
            sampler_type = SAMPLER_TYPES[j]

            # 更新args
            args.num_inference_steps = num_inference_steps
            args.sampler_type = sampler_type

            project.progress(experiment_num, total_experiments, f"Running {sampler_type} with {num_inference_steps} steps")

            result = run_single_experiment(results_save_dir, args, experiment_num, total_experiments)
            if result['success']:
                generation_stats = result['generation_stats']
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

            experiment_results.append(result)

        if args.generate_grid:
            # 生成对比图组模式
            project.info("\n🎨 生成对比图组...")
            output_path = project.generate_comparison_grid_from_existing(results_save_dir, args)
            if output_path:
                project.info(f"✅ 对比图组已生成: {output_path}")
            else:
                project.warning("❌ 对比图组生成失败")


    # # Generate comparison table
    # if experiment_results:
    #     table_data = generate_comparison_table(experiment_results)
    #     format_table_with_ranking(table_data)

    #     results_file = os.path.join(results_save_dir, 'celeba_comparison_results.json')
    #     with open(results_file, 'w') as f:
    #         json.dump(experiment_results, f, indent=2)
    #     project.info(f"\n📊 Results saved to: {results_file}")

    end_time = datetime.now()

    # 生成实验总结报告
    project.info("\n生成实验总结报告...")
    project.generate_experiment_summary(results_save_dir, args, experiment_results, start_time, end_time)

    project.log_experiment_end("CelebA-HQ 批量实验",
                                duration=(end_time - start_time).total_seconds(),
                                results={'total_experiments': total_experiments, 'successful': len([r for r in experiment_results if r['success']])})

    project.info(f"\n实验完成! 结果保存在: {results_save_dir}")
    project.info(f"日志文件: {logger.get_log_file()}")

    # 关闭日志器
    logger.close()
