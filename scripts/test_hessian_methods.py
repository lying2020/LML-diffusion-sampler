#!/usr/bin/env python3
"""
Test Different Hessian Computation Methods for LML

This script compares:
1. Original LML method
2. Explicit Hessian computation
3. Hessian-Free method (CG + HVP)
"""

import sys
import os
import time
import torch
import argparse
import numpy as np
from typing import Dict, List

sys.path.append(os.getcwd())
from diffusers import DDPMPipeline
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_dpmsolver_multistep_lm_advanced import DPMSolverMultistepLMSchedulerAdvanced

import project as project

def test_hessian_methods(model_id: str, test_num: int = 10, device: str = 'cuda'):
    """Test different Hessian computation methods"""

    print("🔬 Testing Different Hessian Computation Methods")
    print("="*60)

    # Load model
    print("Loading model...")
    pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
    pipe.unet.to(device)

    # Test configurations
    methods = {
        'original': 'Original LML (simplified approximation)',
        'explicit': 'Explicit Hessian computation',
        'hessian_free': 'Hessian-Free (CG + HVP)'
    }

    results = {}

    for method_name, method_desc in methods.items():
        print(f"\n{'='*60}")
        print(f"Testing: {method_desc}")
        print(f"{'='*60}")

        # Configure scheduler
        if method_name == 'original':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lamb = 0.0008
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = 1.0e-8
        else:
            pipe.scheduler = DPMSolverMultistepLMSchedulerAdvanced.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lamb = 0.0008
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = 1.0e-8
            pipe.scheduler.hessian_method = method_name
            pipe.scheduler.set_model(pipe.unet)

        # Generate images
        print(f"Generating {test_num} images with {method_name} method...")

        start_time = time.time()
        generation_times = []
        image_qualities = []

        with torch.no_grad():
            for seed in range(test_num):
                torch.manual_seed(seed)

                step_start = time.time()
                images = pipe(batch_size=1, num_inference_steps=20).images
                step_time = time.time() - step_start
                generation_times.append(step_time)

                # Simple quality metric (file size as proxy)
                if images:
                    img_tensor = torch.tensor(np.array(images[0])).float()
                    quality = torch.var(img_tensor).item()  # Variance as quality proxy
                    image_qualities.append(quality)

                if seed % 5 == 0:
                    print(f"  Generated {seed+1}/{test_num} images")

        total_time = time.time() - start_time
        avg_time = np.mean(generation_times)
        avg_quality = np.mean(image_qualities) if image_qualities else 0

        results[method_name] = {
            'total_time': total_time,
            'avg_time_per_image': avg_time,
            'avg_quality': avg_quality,
            'generation_times': generation_times,
            'image_qualities': image_qualities
        }

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per image: {avg_time:.2f}s")
        print(f"  Average quality score: {avg_quality:.4f}")

    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")

    print(f"{'Method':<15} {'Time (s)':<10} {'Quality':<10} {'Speedup':<10}")
    print("-" * 50)

    baseline_time = results['original']['avg_time_per_image']

    for method_name, result in results.items():
        speedup = baseline_time / result['avg_time_per_image']
        print(f"{method_name:<15} {result['avg_time_per_image']:<10.3f} {result['avg_quality']:<10.4f} {speedup:<10.2f}x")

    # Detailed analysis
    print(f"\n{'='*60}")
    print("DETAILED ANALYSIS")
    print(f"{'='*60}")

    for method_name, result in results.items():
        print(f"\n{method_name.upper()} Method:")
        print(f"  Generation times: {np.min(result['generation_times']):.3f}s - {np.max(result['generation_times']):.3f}s")
        print(f"  Quality scores: {np.min(result['image_qualities']):.4f} - {np.max(result['image_qualities']):.4f}")
        print(f"  Time std dev: {np.std(result['generation_times']):.3f}s")
        print(f"  Quality std dev: {np.std(result['image_qualities']):.4f}")

    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")

    fastest_method = min(results.keys(), key=lambda k: results[k]['avg_time_per_image'])
    best_quality_method = max(results.keys(), key=lambda k: results[k]['avg_quality'])

    print(f"Fastest method: {fastest_method} ({results[fastest_method]['avg_time_per_image']:.3f}s per image)")
    print(f"Best quality method: {best_quality_method} ({results[best_quality_method]['avg_quality']:.4f} quality score)")

    if fastest_method == best_quality_method:
        print(f"✅ {fastest_method} provides both speed and quality benefits!")
    else:
        print(f"⚖️  Trade-off between speed ({fastest_method}) and quality ({best_quality_method})")

    print(f"\n�� For production use:")
    print(f"  - Use 'original' for maximum speed")
    print(f"  - Use 'hessian_free' for better quality with reasonable speed")
    print(f"  - Use 'explicit' for research/analysis (slower but more accurate)")

def main():
    parser = argparse.ArgumentParser(description="Test Hessian computation methods")
    parser.add_argument('--model_id', type=str, default='./model/ddpm_ema_cifar10',
                        help='Path to the model')
    parser.add_argument('--test_num', type=int, default=10,
                        help='Number of test images to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Get absolute path
    model_id = os.path.join(project.model_dir, args.model_id)

    test_hessian_methods(model_id, args.test_num, args.device)

if __name__ == '__main__':
    main()
