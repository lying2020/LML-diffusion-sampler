#!/usr/bin/env python3
"""
Unified Stable Diffusion COCO Sampling Script

This script provides a unified interface for COCO image generation using various
diffusion sampling algorithms with Stable Diffusion. It follows the enhanced format
from cifar10.py with comprehensive features and flexible configuration.
"""

# Fix Qt platform plugin issues
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

import sys
import time
import torch
import os
import json
import argparse
import glob
from datetime import datetime

from PIL import Image
import cv2
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, PNDMScheduler, UniPCMultistepScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler
from scheduler.scheduling_dpmsolver_hessian_free import DPMSolverMultistepHessianFreeScheduler

import project as project

#  lambda=0.001 for SD-XL, lambda=0.001 for SD-15
coco_model_path = "/home/liying/Documents/"

"""
Examples:
  # Basic usage with default settings
  python StableDiffusion_COCO.py --sampler_type dpm_lm --test_num 10

  # Generate with specific output directory
  python StableDiffusion_COCO.py --sampler_type ddim --test_num 5 --save_dir ./output/test/coco

  # Use different LML parameters
  python StableDiffusion_COCO.py --sampler_type dpm_lm --lamb 0.001 --kappa 1e-7 --test_num 20

  # Generate with different data type
  python StableDiffusion_COCO.py --sampler_type dpm++ --dtype fp16 --test_num 10
"""

def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="COCO sampling script with enhanced features")

    # Basic parameters
    parser.add_argument('--test_num', type=int, default=2)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=20, choices=[5, 10, 20, 30, 40, 50, 80, 100])

    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--seed', type=int, default=6)

    # Sampler selection
    parser.add_argument('--sampler_type', type=str, default='hessian_free',
                        choices=['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'hessian_free'])
    parser.add_argument('--use_generator', action='store_true', default=True)

    # Output configuration
    parser.add_argument('--save_dir', type=str, default='coco')
    parser.add_argument('--model_path', type=str, default=coco_model_path)
    parser.add_argument('--model_type', type=str, default='stable-diffusion-v1-5', choices=['stable-diffusion-v1-5', 'stable-diffusion-xl-base-1.0', 'stable-diffusion-2-base'])
    parser.add_argument('--coco_prompts_file', type=str, default="coco_top_40_prompts.json", choices=['coco_top_40_prompts.json', 'coco_3w_prompts.json', 'fid_1k_json.json', 'fid_3w_json.json'])

    # LML parameters
    parser.add_argument('--lamb', type=float, default=0.001)
    parser.add_argument('--kappa', type=float, default=1.0e-8)

    # Technical parameters
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp64', 'fp16', 'bf16'])
    parser.add_argument('--device', type=str, default='cuda')

    # Evaluation options
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation metrics')
    parser.add_argument('--save_results', action='store_true', help='Save evaluation results to file')
    parser.add_argument('--generate_grid', action='store_true', default=False, help='Generate comparison grid from existing images')
    parser.add_argument('--grid_title', type=str, default="COCO Generation Comparison", help='Title of comparison grid')
    parser.add_argument('--grid_test_num', type=int, default=6, help='Number of images to test in grid')
    parser.add_argument('--grid_samplers', default=['ddim', 'pndm', 'dpm++', 'dpm', 'unipc', 'hessian_free'],
                        help='List of samplers to test in batch mode')

    # Batch processing options
    parser.add_argument('--run_batch', action='store_true', default=False, help='Run batch experiments with multiple samplers and steps')
    parser.add_argument('--run_batch_samplers', default=['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'hessian_free'], help='List of samplers to test in batch mode')
    parser.add_argument('--run_batch_steps', type=int, default=[10, 20, 50], help='List of inference steps to test in batch mode')

    # Additional options
    parser.add_argument('--save_log', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    return args

def setup_scheduler(pipe, sampler_type, lamb=5.0, kappa=0.0):
    """Setup the appropriate scheduler based on sampler type"""

    if sampler_type == 'pndm':
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        print(f"  Using PNDM scheduler")

    elif sampler_type == 'ddim':
        pipe.scheduler = DDIMLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = False
        pipe.scheduler.kappa = kappa
        print(f"  Using DDIM scheduler")

    elif sampler_type == 'ddim_lm':
        pipe.scheduler = DDIMLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = True
        pipe.scheduler.kappa = kappa
        print(f"  Using DDIM with LML correction (λ={lamb}, κ={kappa})")

    elif sampler_type == 'dpm++':
        pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.config.algorithm_type = "dpmsolver++"
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = False
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
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = False
        print(f"  Using DPM-Solver scheduler")

    elif sampler_type == 'unipc':
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        print(f"  Using UniPC scheduler")

    elif sampler_type == 'hessian_free':
        pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.config.algorithm_type = "dpmsolver"
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = True
        pipe.scheduler.kappa = kappa
        print(f"  Using DPM-Solver with LML correction (λ={lamb}, κ={kappa})")

        # pipe.scheduler = DPMSolverMultistepHessianFreeScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler.config.solver_order = 3
        # pipe.scheduler.config.algorithm_type = "dpmsolver++"
        # pipe.scheduler.lamb = lamb
        # pipe.scheduler.lm = True
        # pipe.scheduler.kappa = kappa
        # pipe.scheduler.hessian_method = 'hessian_free'
        # # 设置模型用于Hessian计算
        # pipe.scheduler.set_model(pipe.unet)
        # print(f"  Using DPM-Solver++ with Hessian-Free LML correction (λ={lamb}, κ={kappa})")

    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

def load_coco_prompts(coco_prompts_path):
    """Load COCO prompts from JSON file"""

    try:
        with open(coco_prompts_path) as fr:
            COCO_prompts_dict = json.load(fr)
        return COCO_prompts_dict
    except FileNotFoundError:
        print("⚠️  COCO prompts file not found. Using default prompts.")
        # Fallback prompts for testing
        return {
            "00001": "a beautiful landscape with mountains and trees",
            "00002": "a cat sitting on a windowsill",
            "00003": "a modern city skyline at sunset",
            "00004": "a vintage car parked on the street",
            "00005": "a delicious plate of pasta"
        }

def generate_images(results_save_dir, args, pipe):
    """Generate images using the specified pipeline"""

    total_time = 0
    generation_times = []

    # Load COCO prompts
    # coco_prompts_path = os.path.join(project.project_dir, "evaluation", "coco_prompts", "COCO_3W_prompt.json")
    coco_prompts_path = os.path.join(project.project_dir, "evaluations", "coco_prompts", args.coco_prompts_file)
    COCO_prompts_dict = load_coco_prompts(coco_prompts_path)
    image_ids = list(COCO_prompts_dict.keys())

    print(f"\n{'='*60}")
    print(f"Starting COCO image generation")
    print(f"{'='*60}")
    print(f"Sampler: {args.sampler_type} - {project.get_sampler_description(args.sampler_type)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance}")
    print(f"Total images: {args.test_num}")
    print(f"Save directory: {results_save_dir}")
    print(f"{'='*60}")

    generated_count = 0
    for pi, key in enumerate(image_ids):
        if pi >= args.start_index and pi < args.start_index + args.test_num:
            print(f"\nGenerating image {pi - args.start_index + 1}/{args.test_num} (ID: {key})")
            print(f"Prompt: {COCO_prompts_dict[key]}")

            batch_start_time = time.time()
            prompt = COCO_prompts_dict[key]
            negative_prompt = None

            for seed in [args.seed]:  # Can be extended to multiple seeds
                generator = torch.Generator(device='cuda')
                generator = generator.manual_seed(seed)

                # Generate image
                with torch.no_grad():
                    res = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=args.guidance,
                        generator=generator
                    ).images[0]

                # Save image
                filename = f"{pi:05d}_{key}_guidance{args.guidance}_inference{args.num_inference_steps}_seed{seed}_{args.sampler_type}.jpg"
                filepath = os.path.join(results_save_dir, filename)
                res.save(filepath)
                generated_count += 1

            batch_time = time.time() - batch_start_time
            generation_times.append(batch_time)
            total_time += batch_time

            print(f"  ✓ Generated image in {batch_time:.3f}s")
            print(f"  ✓ Saved to: {filepath}")
            print(f"{sampler_type}##{key},done")

    # Print summary
    avg_time = total_time / args.test_num if args.test_num > 0 else 0

    print(f"\n{'='*60}")
    print(f"GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {avg_time:.3f}s")
    print(f"Total images generated: {generated_count}")
    print(f"Images per second: {generated_count / total_time:.2f}" if total_time > 0 else "N/A")
    print(f"{'='*60}")

    return {
        'total_time': total_time,
        'avg_time_per_image': avg_time,
        'total_images': generated_count,
        'images_per_second': generated_count / total_time if total_time > 0 else 0,
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

        # Setup paths
        model_path = os.path.join(args.model_path, args.model_type)

        # Handle results_save_dir path properly
        results_save_dir = os.path.join(results_save_dir, "steps"+'_'+str(args.num_inference_steps), args.sampler_type)
        os.makedirs(results_save_dir, exist_ok=True)

        print(f"🚀 Stable Diffusion COCO Sampling Script")
        print("="*60)
        print(f"Model: {model_path}")
        print(f"Device: {args.device}")
        print(f"Data type: {args.dtype}")
        print(f"Sampler: {args.sampler_type}")
        print(f"Output: {results_save_dir}")
        print("="*60)

        # Load pipeline
        print("\n📦 Loading model...")

        # Choose the appropriate pipeline based on model type
        if args.model_type == 'stable-diffusion-xl-base-1.0':
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                safety_checker=None,
                added_cond_kwargs={}
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                safety_checker=None
            )

        pipe = pipe.to(args.device)
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
        SAMPLER_TYPES = args.run_batch_samplers  #['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'hessian_free']
        INFERENCE_STEPS = args.run_batch_steps  # [5, 10, 20, 30]

    total_experiments = len(SAMPLER_TYPES) * len(INFERENCE_STEPS)
    experiment_results = []
    start_time = datetime.now()
    experiment_num = 0

    print("="*50)
    print("Stable Diffusion COCO 实验批量运行开始")
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
    print("Stable Diffusion COCO 实验批量运行完成")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总实验数: {total_experiments}")
    print("="*50)

    # 生成实验总结报告
    print("\n生成实验总结报告...")
    project.generate_experiment_summary(results_save_dir, args, experiment_results, start_time, end_time)

    print(f"\n实验完成! 结果保存在: {results_save_dir}")
