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
from datetime import datetime

sys.path.append(os.getcwd())
from diffusers import DDPMPipeline, DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler

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
    parser.add_argument('--test_num', type=int, default=64)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_inference_steps', type=int, default=20)

    # Sampler selection
    parser.add_argument('--sampler_type', type=str, default='pndm',
                        choices=['pndm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc'])

    # Output configuration
    parser.add_argument('--save_dir', type=str, default='cifar10')
    parser.add_argument('--model_id', type=str, default='ddpm_ema_cifar10')

    # LML parameters
    parser.add_argument('--lamb', type=float, default=0.0008)
    parser.add_argument('--kappa', type=float, default=1.0e-8)

    # Technical parameters
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp64', 'fp16', 'bf16'])
    parser.add_argument('--device', type=str, default='cuda')

    # Additional options
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    return args

def get_sampler_description(sampler_type):
    """Get description for different sampler types"""
    descriptions = {
        'ddim': 'Denoising Diffusion Implicit Models - Deterministic sampling',
        'dpm': 'DPM-Solver - High-order solver for diffusion ODEs',
        'dpm++': 'DPM-Solver++ - Improved version with better stability',
        'dpm_lm': 'DPM-Solver with Levenberg-Marquardt Langevin correction',
        'pndm': 'Pseudo Numerical methods for Diffusion Models',
        'unipc': 'Unified Predictor-Corrector framework'
    }
    return descriptions.get(sampler_type, 'Unknown sampler type')

def setup_scheduler(pipe, sampler_type, lamb=0.0008, kappa=1e-8):
    """Setup the appropriate scheduler based on sampler type"""

    if sampler_type == 'pndm':
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
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
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        print(f"  Using UniPC scheduler")

    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

def generate_images(pipe, batch_size, num_inference_steps, test_num, start_index, save_dir, sampler_type):
    """Generate images using the specified pipeline"""

    total_time = 0
    generation_times = []

    print(f"\n{'='*60}")
    print(f"Starting image generation")
    print(f"{'='*60}")
    print(f"Sampler: {sampler_type} - {get_sampler_description(sampler_type)}")
    print(f"Batch size: {batch_size}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Total images: {test_num}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*60}")

    for seed in range(start_index, start_index + test_num):
        print(f"\nGenerating batch {seed - start_index + 1}/{test_num} (seed={seed})")
        batch_start_time = time.time()
        torch.manual_seed(seed)

        # Generate images
        with torch.no_grad():
            images = pipe(batch_size=batch_size, num_inference_steps=num_inference_steps).images

        # Save images
        for i, image in enumerate(images):
            filename = f"cifar10_{sampler_type}_inference{num_inference_steps}_seed{seed}_{i}.png"
            filepath = os.path.join(save_dir, filename)
            image.save(filepath)

        batch_time = time.time() - batch_start_time
        generation_times.append(batch_time)
        total_time += batch_time

        print(f"  ✓ Generated {len(images)} images in {batch_time:.3f}s")
        print(f"  ✓ Saved to: {save_dir}")

    # Print summary
    avg_time = total_time / test_num
    avg_time_per_image = avg_time / batch_size

    print(f"\n{'='*60}")
    print(f"GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per batch: {avg_time:.3f}s")
    print(f"Average time per image: {avg_time_per_image:.3f}s")
    print(f"Total images generated: {test_num * batch_size}")
    print(f"Images per second: {test_num * batch_size / total_time:.2f}")
    print(f"{'='*60}")

    return {
        'total_time': total_time,
        'avg_time_per_batch': avg_time,
        'avg_time_per_image': avg_time_per_image,
        'total_images': test_num * batch_size,
        'images_per_second': test_num * batch_size / total_time,
        'generation_times': generation_times
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

    print(f"  ✓ Generation log saved to: {log_path}")

def main():
    args = parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        'fp32': torch.float32,
        'fp64': torch.float64,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]

    # Setup paths - fix the path handling
    model_id = os.path.join(project.model_dir, args.model_id)

    # Handle save_dir path properly
    save_dir = os.path.join(project.output_dir, args.save_dir, args.sampler_type)
    os.makedirs(save_dir, exist_ok=True)

    print("🚀 CIFAR-10 Unified Sampling Script")
    print("="*60)
    print(f"Model: {model_id}")
    print(f"Device: {args.device}")
    print(f"Data type: {args.dtype}")
    print(f"Sampler: {args.sampler_type}")
    print(f"Output: {save_dir}")
    print("="*60)

    try:
        # Load pipeline
        print("\n📦 Loading model...")
        pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=False)
        pipe.unet.to(args.device)
        print("  ✓ Model loaded successfully")

        # Setup scheduler
        print(f"\n⚙️ Setting up scheduler...")
        setup_scheduler(pipe, args.sampler_type, args.lamb, args.kappa)

        # Generate images
        generation_stats = generate_images(
            pipe, args.batch_size, args.num_inference_steps,
            args.test_num, args.start_index, save_dir, args.sampler_type
        )

        # Save generation log if requested
        if args.save_log:
            print(f"\n📝 Saving generation log...")
            save_generation_log(save_dir, args.sampler_type, generation_stats, args)

        print(f"\n✅ Generation completed successfully!")
        print(f"   Generated {generation_stats['total_images']} images")
        print(f"   Total time: {generation_stats['total_time']:.2f}s")
        print(f"   Average time per image: {generation_stats['avg_time_per_image']:.3f}s")

    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
