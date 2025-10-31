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
from evaluations.celeba_eva import evaluate_images
import project as project

celeba_model_path = "/home/liying/Documents/ldm-celebahq-256/"
# celeba_model_path = "/research/cbim/vast/cj574/data/diffusion/celeba_hq_256/ldm-celebahq-256"


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="CelebA-HQ sampling script with enhanced features")

    # Basic parameters
    parser.add_argument('--test_num', type=int, default=2)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=40, choices=[5, 10, 20, 40, 70, 100, 200, 400, 600, 1000])

    parser.add_argument('--scaling_factor', type=float, default=0.18215)
    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--seed', type=int, default=6)

    # Sampler selection
    parser.add_argument('--sampler_type', type=str, default='dpm++',
                        choices=['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'hcg'])
    parser.add_argument('--use_generator', action='store_true', default=True)

    # Output configuration
    parser.add_argument('--save_dir', type=str, default='celebahq_256')
    parser.add_argument('--model_path', type=str, default=celeba_model_path)
    parser.add_argument('--model_type', type=str, default="ldm")

    # LML parameters
    parser.add_argument('--lamb', type=float, default=0.004)
    parser.add_argument('--kappa', type=float, default=1.0e-8)

    # HCG (Hessian-Conjugate Gradient) parameters
    parser.add_argument('--kappa_target', type=float, default=10.0, help='Target condition number for adaptive damping')
    parser.add_argument('--lanczos_k', type=int, default=10, help='Number of Lanczos iterations for eigenvalue estimation')
    parser.add_argument('--cg_max_iter', type=int, default=20, help='Maximum CG iterations')
    parser.add_argument('--cg_tol', type=float, default=1e-4, help='CG tolerance')
    parser.add_argument('--use_spectral_scaling', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False, help='Use spectral radius scaling (default: True)')

    # Technical parameters
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp64', 'fp16', 'bf16'])
    parser.add_argument('--device', type=str, default='cuda')

    # Evaluation options
    parser.add_argument('--evaluate', action='store_true', default=False, help='Run evaluation metrics')
    parser.add_argument('--generate_grid', action='store_true', default=False, help='Generate comparison grid from existing images')
    parser.add_argument('--grid_title', type=str, default="CelebA-HQ Generation Comparison", help='Title of comparison grid')
    parser.add_argument('--grid_test_num', type=int, default=6, help='Number of images to test in grid')
    parser.add_argument('--grid_test_index', type=list, default=[0, 1, 2, 3, 4, 5], help='Index of images to test in grid')
    parser.add_argument('--grid_samplers', default=['ddim', 'pndm', 'dpm++', 'dpm', 'unipc', 'hcg'],
                        help='List of samplers to test in batch mode')

    # Batch processing options
    parser.add_argument('--run_batch', action='store_true', default=False, help='Run batch experiments with multiple samplers and steps')
    parser.add_argument('--run_batch_samplers', default=['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'hcg'], help='List of samplers to test in batch mode')
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

    if sampler_type == 'pndm':
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        project.info(f"  Using PNDM scheduler")

    elif sampler_type == 'ddim':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.eta = 0.0  # 设置eta=0.0，与celeba_test.py保持一致
        project.info(f"  Using DDIM scheduler (eta=0.0)")

    elif sampler_type == 'ddim_lm':
        pipe.scheduler = DDIMLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = True
        pipe.scheduler.kappa = kappa
        project.info(f"  Using DDIM with LML correction (λ={lamb}, κ={kappa})")

    elif sampler_type == 'dpm++':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.algorithm_type = "dpmsolver++"
        pipe.scheduler.config.solver_order = 3
        project.info(f"  Using DPM-Solver++ scheduler")

    elif sampler_type == 'dpm_lm':
        pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.algorithm_type = "dpmsolver"
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = True
        pipe.scheduler.kappa = kappa
        project.info(f"  Using DPM-Solver with LML correction (λ={lamb}, κ={kappa})")

    elif sampler_type == 'dpm':
        pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.algorithm_type = "dpmsolver"
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.lm = False
        project.info(f"  Using DPM-Solver scheduler")

    elif sampler_type == 'unipc':
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        project.info(f"  Using UniPC scheduler")

    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

def setup_scheduler_hcg(pipe, kappa_target=10.0, lanczos_k=10, cg_max_iter=20,
                    cg_tol=1e-4, use_spectral_scaling=True):
    """Setup the HCG scheduler"""
    # 使用新的 DPMSolverMultistepHCGScheduler
    pipe.scheduler = DPMSolverMultistepHCGScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.config.algorithm_type = "dpmsolver++"
    pipe.scheduler.config.solver_order = 3

    # 设置模型用于Hessian计算（必需）
    # pipe.scheduler.set_model(pipe.unet)

    # 设置HCG参数
    pipe.scheduler.use_hcg = True
    pipe.scheduler.kappa_target = kappa_target
    pipe.scheduler.lanczos_k = lanczos_k
    pipe.scheduler.cg_max_iter = cg_max_iter
    pipe.scheduler.cg_tol = cg_tol
    pipe.scheduler.use_spectral_scaling = use_spectral_scaling

    project.info(f"  Using DPM-Solver++ with HCG (Hessian-Conjugate Gradient) correction")
    project.info(f"    kappa_target={kappa_target}, lanczos_k={lanczos_k}, cg_max_iter={cg_max_iter}")
    project.info(f"    cg_tol={cg_tol}, use_spectral_scaling={use_spectral_scaling}")

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
        if args.sampler_type == 'hcg' and pipe.scheduler.model is not None:
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
    if args.sampler_type == 'hcg':
        setup_scheduler_hcg(
            pipe, kappa_target=args.kappa_target,
            lanczos_k=args.lanczos_k,
            cg_max_iter=args.cg_max_iter,
            cg_tol=args.cg_tol,
            use_spectral_scaling=args.use_spectral_scaling
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
    # 设置日志系统
    logger = project.setup_logging(name='celeba', level=project.logging.INFO)

    args = parse_args()
    results_save_dir = os.path.join(project.output_dir, args.save_dir + '_' + args.model_type)

    # 单个实验模式（保持原有逻辑）
    SAMPLER_TYPES = [args.sampler_type]
    INFERENCE_STEPS = [args.num_inference_steps]

    # 检查是否运行批量实验
    if hasattr(args, 'run_batch') and args.run_batch:
        # 批量实验模式
        SAMPLER_TYPES = args.run_batch_samplers  #['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'hcg']
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

    print("="*50)
    print("CelebA-HQ 实验批量运行开始")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

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
            print("\n🎨 生成对比图组...")
            output_path = project.generate_comparison_grid_from_existing(results_save_dir, args)
            if output_path:
                print(f"✅ 对比图组已生成: {output_path}")
            else:
                print("❌ 对比图组生成失败")


    # # Generate comparison table
    # if experiment_results:
    #     table_data = generate_comparison_table(experiment_results)
    #     format_table_with_ranking(table_data)

    #     results_file = os.path.join(results_save_dir, 'celeba_comparison_results.json')
    #     with open(results_file, 'w') as f:
    #         json.dump(experiment_results, f, indent=2)
    #     project.info(f"\n📊 Results saved to: {results_file}")

    end_time = datetime.now()

    print("\n" + "="*50)
    print("CelebA-HQ 实验批量运行完成")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

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
