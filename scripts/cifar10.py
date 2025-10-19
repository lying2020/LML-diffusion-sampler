#!/usr/bin/env python3
"""
Unified CIFAR-10 Sampling Script

This script provides a unified interface for CIFAR-10 image generation using various
diffusion sampling algorithms. It combines the functionality of cifar10.py and
cifar10_updated.py with enhanced features and flexible configuration.
"""

import sys
import time
import torch
import os
import json
import argparse
import glob
from datetime import datetime

sys.path.append(os.getcwd())
from diffusers import DDPMPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_unipc_multistep_lm import UniPCMultistepSchedulerLM
from scheduler.scheduling_ddim_lm import DDIMLMScheduler
from scheduler.scheduling_pndm_lm import PNDMSchedulerLM
from scheduler.scheduling_dpmsolver_hessian_free import DPMSolverMultistepHessianFreeScheduler

import project as project


"""
Examples:
  # Basic usage with default settings
  python cifar10.py --sampler_type dpm_lm --test_num 10

  # Generate with specific output directory
  python cifar10.py --sampler_type ddim --test_num 5 --save_dir ./output/test/cifar10

  # Use different LML parameters
  python cifar10.py --sampler_type dpm_lm --lamb 0.001 --kappa 1e-7 --test_num 20

  # Generate with different data type
  python cifar10.py --sampler_type dpm++ --dtype fp16 --test_num 10
"""

def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="CIFAR-10 sampling script with enhanced features")

    # Basic parameters
    parser.add_argument('--test_num', type=int, default=20)
    parser.add_argument('--start_index', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=20)

    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--seed', type=int, default=6)

    # Sampler selection
    parser.add_argument('--sampler_type', type=str, default='hessian_free',
                        choices=['pndm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'hessian_free'])
    parser.add_argument('--use_generator', action='store_true', default=True)

    # Output configuration
    parser.add_argument('--save_dir', type=str, default='cifar10')
    parser.add_argument('--model_path', type=str, default='ddpm_ema_cifar10')
    parser.add_argument('--model_type', type=str, default="ddpm")

    # LML parameters
    parser.add_argument('--lamb', type=float, default=0.0008)
    parser.add_argument('--kappa', type=float, default=1.0e-8)

    # Technical parameters
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp64', 'fp16', 'bf16'])
    parser.add_argument('--device', type=str, default='cuda')

    # Evaluation options
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation metrics')
    parser.add_argument('--save_results', action='store_true', help='Save evaluation results to file')
    parser.add_argument('--generate_grid', action='store_true', default=True, help='Generate comparison grid from existing images')
    parser.add_argument('--grid_title', type=str, default="CIFAR-10 Generation Comparison", help='Title of comparison grid')
    parser.add_argument('--grid_test_num', type=int, default=6, help='Number of images to test in grid')
    parser.add_argument('--grid_samplers', default=['ddim', 'pndm', 'dpm++', 'dpm', 'unipc', 'hessian_free'],
                        help='List of samplers to test in batch mode')

    # Batch processing options
    parser.add_argument('--run_batch', action='store_true', default=True, help='Run batch experiments with multiple samplers and steps')
    parser.add_argument('--run_batch_samplers', default=['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'hessian_free'], help='List of samplers to test in batch mode')
    parser.add_argument('--run_batch_steps', type=int, default=[10, 20, 50], help='List of inference steps to test in batch mode')

    # Additional options
    parser.add_argument('--save_log', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    return args

def setup_scheduler(pipe, sampler_type, lamb=0.0008, kappa=1e-8):
    """Setup the appropriate scheduler based on sampler type"""

    if sampler_type == 'pndm':
        pipe.scheduler = PNDMSchedulerLM.from_config(pipe.scheduler.config)
        print(f"  Using PNDM scheduler")

    elif sampler_type == 'ddim':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        print(f"  Using DDIM scheduler")

    elif sampler_type == 'dpm++':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.config.algorithm_type = "dpmsolver++"
        print(f"  Using DPM-Solver++ scheduler")

    elif sampler_type == 'dpm_lm':
        pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.config.algorithm_type = "dpmsolver"
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = True
        pipe.scheduler.kappa = kappa
        print(f"  Using DPM-Solver with LML correction (λ={lamb}, κ={kappa})")

    elif sampler_type == 'dpm':
        pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.config.algorithm_type = "dpmsolver"
        pipe.scheduler.lm = False
        print(f"  Using DPM-Solver scheduler")

    elif sampler_type == 'unipc':
        pipe.scheduler = UniPCMultistepSchedulerLM.from_config(pipe.scheduler.config)
        print(f"  Using UniPC scheduler")

    elif sampler_type == 'hessian_free':
        pipe.scheduler = DPMSolverMultistepHessianFreeScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.config.algorithm_type = "dpmsolver++"
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = True
        pipe.scheduler.kappa = kappa
        pipe.scheduler.hessian_method = 'hessian_free'
        # 设置模型用于Hessian计算
        pipe.scheduler.set_model(pipe.unet)
        print(f"  Using DPM-Solver++ with Hessian-Free LML correction (λ={lamb}, κ={kappa})")

    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

def generate_images(results_save_dir, args, pipe):
    """Generate images using the specified pipeline"""

    total_time = 0
    generation_times = []

    print(f"\n{'='*60}")
    print(f"Starting image generation")
    print(f"{'='*60}")
    print(f"Sampler: {args.sampler_type} - {project.get_sampler_description(args.sampler_type)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Total images: {args.test_num}")
    print(f"Save directory: {results_save_dir}")
    print(f"{'='*60}")

    for seed in range(args.seed, args.seed + args.test_num):
        print(f"\nGenerating batch {seed - args.seed + 1}/{args.test_num} (seed={seed})")
        batch_start_time = time.time()
        torch.manual_seed(seed)
        if args.use_generator:
            generator = torch.Generator(device='cuda').manual_seed(seed)
        else:
            generator = None
        # Generate images
        with torch.no_grad():
            images = pipe(batch_size=args.batch_size, num_inference_steps=args.num_inference_steps, generator=generator).images

        # Save images
        for i, image in enumerate(images):
            filename = f"cifar10_{args.sampler_type}_inference{args.num_inference_steps}_seed{seed}_{i}.png"
            filepath = os.path.join(results_save_dir, filename)
            image.save(filepath)

        batch_time = time.time() - batch_start_time
        generation_times.append(batch_time)
        total_time += batch_time

        print(f"  ✓ Generated {len(images)} images in {batch_time:.3f}s")
        print(f"  ✓ Saved to: {results_save_dir}")

    # Print summary
    avg_time = total_time / args.test_num
    avg_time_per_image = avg_time / args.batch_size

    print(f"\n{'='*60}")
    print(f"GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per batch: {avg_time:.3f}s")
    print(f"Average time per image: {avg_time_per_image:.3f}s")
    print(f"Total images generated: {args.test_num * args.batch_size}")
    print(f"Images per second: {args.test_num * args.batch_size / total_time:.2f}")
    print(f"{'='*60}")

    return {
        'total_time': total_time,
        'avg_time_per_batch': avg_time,
        'avg_time_per_image': avg_time_per_image,
        'total_images': args.test_num * args.batch_size,
        'images_per_second': args.test_num * args.batch_size / total_time,
        'generation_times': generation_times
    }

def run_single_experiment(results_save_dir, args, experiment_num, total_experiments):
    """Run a single experiment with enhanced logging"""

    print("")
    print("="*50)
    print(f"实验 {experiment_num}/{total_experiments}")
    print(f"Sampler: {args.sampler_type}")
    print(f"Steps: {args.num_inference_steps}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

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

        # Setup paths - fix the path handling
        model_path = os.path.join(project.model_dir, args.model_path)

        # Handle results_save_dir path properly
        results_save_dir = os.path.join(results_save_dir, "steps"+'_'+str(args.num_inference_steps), args.sampler_type)
        os.makedirs(results_save_dir, exist_ok=True)

        print(f"🚀 CIFAR-10 Unified Sampling Script")
        print("="*60)
        print(f"Model: {model_path}")
        print(f"Device: {args.device}")
        print(f"Data type: {args.dtype}")
        print(f"Sampler: {sampler_type}")
        print(f"Output: {results_save_dir}")
        print("="*60)

        # Load pipeline
        print("\n📦 Loading model...")
        pipe = DDPMPipeline.from_pretrained(model_path, torch_dtype=dtype, use_safetensors=False)
        pipe.unet.to(args.device)
        print("  ✓ Model loaded successfully")

        # Setup scheduler
        print(f"\n⚙️ Setting up scheduler...")
        setup_scheduler(pipe, args.sampler_type, args.lamb, args.kappa)

        # Generate images
        generation_stats = generate_images(results_save_dir, args, pipe)

        # Save generation log if requested
        if args.save_log:
            print(f"\n📝 Saving generation log...")
            project.save_generation_log(results_save_dir, args, generation_stats)

        # 计算实验耗时
        exp_end_time = time.time()
        exp_duration = exp_end_time - exp_start_time

        # 统计生成的图片数量
        image_count = project.count_generated_images(results_save_dir)

        print(f"\n✅ 实验完成! 耗时: {exp_duration:.1f}秒")
        print(f"生成图片数量: {image_count}")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            'success': True,
            'duration': exp_duration,
            'image_count': image_count,
            'generation_stats': generation_stats,
            'results_save_dir': results_save_dir
        }

    except Exception as e:
        exp_end_time = time.time()
        exp_duration = exp_end_time - exp_start_time

        print(f"\n❌ 实验失败! 耗时: {exp_duration:.1f}秒")
        print(f"错误信息: {str(e)}")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            'success': False,
            'duration': exp_duration,
            'error': str(e),
            'sampler_type': args.sampler_type,
            'num_inference_steps': args.num_inference_steps
        }


if __name__ == '__main__':
    args = parse_args()
    results_save_dir = os.path.join(project.output_dir, args.save_dir + '_' + args.model_type)

    # 单个实验模式（保持原有逻辑）
    SAMPLER_TYPES = [args.sampler_type]
    INFERENCE_STEPS = [args.num_inference_steps]

    # 检查是否运行批量实验
    if hasattr(args, 'run_batch') and args.run_batch:
        # 批量实验模式
        SAMPLER_TYPES = args.run_batch_samplers    #['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'hessian_free']
        INFERENCE_STEPS = args.run_batch_steps #[5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 80]

    total_experiments = len(SAMPLER_TYPES) * len(INFERENCE_STEPS)
    experiment_results = []
    start_time = datetime.now()
    experiment_num = 0

    print("="*50)
    print("CIFAR-10 实验批量运行开始")
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

            result = run_single_experiment(results_save_dir, args, experiment_num, total_experiments)
            experiment_results.append(result)

        if args.generate_grid:
            # 生成对比图组模式
            print("\n🎨 生成对比图组...")
            output_path = project.generate_comparison_grid_from_existing(
                results_save_dir, args.num_inference_steps, args.grid_test_num, args.grid_samplers, args.grid_title)
            if output_path:
                print(f"✅ 对比图组已生成: {output_path}")
            else:
                print("❌ 对比图组生成失败")

    end_time = datetime.now()

    print("\n" + "="*50)
    print("CIFAR-10 实验批量运行完成")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总实验数: {total_experiments}")
    print("="*50)

    # 生成实验总结报告
    print("\n生成实验总结报告...")
    project.generate_experiment_summary(results_save_dir, args, experiment_results, start_time, end_time)

    print(f"\n实验完成! 结果保存在: {results_save_dir}")
