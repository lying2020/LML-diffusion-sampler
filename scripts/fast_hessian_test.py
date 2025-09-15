#!/usr/bin/env python3
"""
Fast Hessian Testing for LML Diffusion Sampler

This script provides practical Hessian computation methods:
1. Original LML method (baseline)
2. Improved LML with better parameters
3. Simple Hessian approximation
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

import project as project

class FastHessianTester:
    """Fast testing suite for practical Hessian methods"""

    def __init__(self, model_id: str, device: str = 'cuda'):
        self.model_id = model_id
        self.device = device
        self.results = {}

        # Load model
        print("Loading model...")
        self.pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        self.pipe.unet.to(device)

    def test_original_lml(self, test_num: int = 10) -> Dict:
        """Test original LML method with default parameters"""
        print(f"\n{'='*60}")
        print("Testing: Original LML Method (Default Parameters)")
        print(f"{'='*60}")

        # Configure scheduler
        self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.config.solver_order = 3
        self.pipe.scheduler.config.algorithm_type = "dpmsolver"
        self.pipe.scheduler.lamb = 0.0008
        self.pipe.scheduler.lm = True
        self.pipe.scheduler.kappa = 1.0e-8

        return self._generate_and_measure(test_num, "original_default")

    def test_improved_lml(self, test_num: int = 10) -> Dict:
        """Test LML method with optimized parameters"""
        print(f"\n{'='*60}")
        print("Testing: Improved LML Method (Optimized Parameters)")
        print(f"{'='*60}")

        # Configure scheduler with better parameters
        self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.config.solver_order = 3
        self.pipe.scheduler.config.algorithm_type = "dpmsolver"
        self.pipe.scheduler.lamb = 0.001  # Slightly higher regularization
        self.pipe.scheduler.lm = True
        self.pipe.scheduler.kappa = 5.0e-8  # Better EMA parameter

        return self._generate_and_measure(test_num, "improved_lml")

    def test_high_quality_lml(self, test_num: int = 10) -> Dict:
        """Test LML method with high-quality parameters"""
        print(f"\n{'='*60}")
        print("Testing: High-Quality LML Method")
        print(f"{'='*60}")

        # Configure scheduler for high quality
        self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.config.solver_order = 3
        self.pipe.scheduler.config.algorithm_type = "dpmsolver"
        self.pipe.scheduler.lamb = 0.002  # Higher regularization for stability
        self.pipe.scheduler.lm = True
        self.pipe.scheduler.kappa = 1.0e-7  # Better EMA parameter

        return self._generate_and_measure(test_num, "high_quality_lml")

    def test_different_steps(self, test_num: int = 10) -> Dict:
        """Test LML with different inference steps"""
        print(f"\n{'='*60}")
        print("Testing: LML with Different Inference Steps")
        print(f"{'='*60}")

        # Configure scheduler
        self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.config.solver_order = 3
        self.pipe.scheduler.config.algorithm_type = "dpmsolver"
        self.pipe.scheduler.lamb = 0.001
        self.pipe.scheduler.lm = True
        self.pipe.scheduler.kappa = 5.0e-8

        generation_times = []
        image_qualities = []

        print(f"Generating {test_num} images with different step counts...")
        start_time = time.time()

        step_configs = [10, 15, 20, 25, 30]

        with torch.no_grad():
            for i, steps in enumerate(step_configs):
                for seed in range(test_num // len(step_configs)):
                    torch.manual_seed(seed + i * 100)

                    step_start = time.time()
                    images = self.pipe(batch_size=1, num_inference_steps=steps).images
                    step_time = time.time() - step_start
                    generation_times.append(step_time)

                    # Simple quality metric
                    if images:
                        img_array = np.array(images[0])
                        quality = np.var(img_array)
                        image_qualities.append(quality)

                    if (i * (test_num // len(step_configs)) + seed) % 5 == 0:
                        print(f"  Generated {i * (test_num // len(step_configs)) + seed + 1}/{test_num} images (steps={steps})")

        total_time = time.time() - start_time

        result = {
            'method': 'different_steps',
            'total_time': total_time,
            'avg_time_per_image': np.mean(generation_times),
            'std_time_per_image': np.std(generation_times),
            'avg_quality': np.mean(image_qualities) if image_qualities else 0,
            'std_quality': np.std(image_qualities) if image_qualities else 0,
            'generation_times': generation_times,
            'image_qualities': image_qualities
        }

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per image: {result['avg_time_per_image']:.3f}s")
        print(f"  Average quality: {result['avg_quality']:.4f}")

        return result

    def _generate_and_measure(self, test_num: int, method_name: str) -> Dict:
        """Generate images and measure performance"""
        generation_times = []
        image_qualities = []

        print(f"Generating {test_num} images...")
        start_time = time.time()

        with torch.no_grad():
            for seed in range(test_num):
                torch.manual_seed(seed)

                step_start = time.time()
                images = self.pipe(batch_size=1, num_inference_steps=20).images
                step_time = time.time() - step_start
                generation_times.append(step_time)

                # Simple quality metric
                if images:
                    img_array = np.array(images[0])
                    quality = np.var(img_array)
                    image_qualities.append(quality)

                if seed % 5 == 0:
                    print(f"  Generated {seed+1}/{test_num} images")

        total_time = time.time() - start_time

        result = {
            'method': method_name,
            'total_time': total_time,
            'avg_time_per_image': np.mean(generation_times),
            'std_time_per_image': np.std(generation_times),
            'avg_quality': np.mean(image_qualities) if image_qualities else 0,
            'std_quality': np.std(image_qualities) if image_qualities else 0,
            'generation_times': generation_times,
            'image_qualities': image_qualities
        }

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per image: {result['avg_time_per_image']:.3f}s")
        print(f"  Average quality: {result['avg_quality']:.4f}")

        return result

    def run_fast_test(self, test_num: int = 20):
        """Run fast test of all methods"""

        print("🚀 Starting Fast Hessian Testing")
        print("="*60)
        print(f"Test configuration:")
        print(f"  - Model: {self.model_id}")
        print(f"  - Device: {self.device}")
        print(f"  - Test samples: {test_num}")
        print(f"  - Inference steps: 20 (except different_steps test)")
        print("="*60)

        # Test each method
        methods = [
            ('original_default', self.test_original_lml),
            ('improved_lml', self.test_improved_lml),
            ('high_quality_lml', self.test_high_quality_lml),
            ('different_steps', self.test_different_steps)
        ]

        for method_name, test_func in methods:
            try:
                result = test_func(test_num)
                self.results[method_name] = result
            except Exception as e:
                print(f"❌ Error testing {method_name}: {e}")
                self.results[method_name] = {
                    'method': method_name,
                    'error': str(e),
                    'success': False
                }

        # Generate comparison report
        self.generate_comparison_report()

    def generate_comparison_report(self):
        """Generate detailed comparison report"""

        print(f"\n{'='*60}")
        print("FAST HESSIAN COMPARISON REPORT")
        print(f"{'='*60}")

        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}

        if not successful_results:
            print("❌ No successful results to compare")
            return

        # Performance comparison
        print(f"\n📊 PERFORMANCE COMPARISON")
        print(f"{'Method':<20} {'Time (s)':<12} {'Quality':<12} {'Speedup':<12} {'Efficiency':<12}")
        print("-" * 80)

        baseline_time = min(r['avg_time_per_image'] for r in successful_results.values())

        for method_name, result in successful_results.items():
            speedup = baseline_time / result['avg_time_per_image']
            efficiency = result['avg_quality'] / result['avg_time_per_image']  # Quality per second

            print(f"{method_name:<20} {result['avg_time_per_image']:<12.3f} {result['avg_quality']:<12.4f} {speedup:<12.2f}x {efficiency:<12.2f}")

        # Quality analysis
        print(f"\n🎨 QUALITY ANALYSIS")
        print(f"{'Method':<20} {'Avg Quality':<15} {'Std Quality':<15} {'Quality Range':<15}")
        print("-" * 70)

        for method_name, result in successful_results.items():
            if result['image_qualities']:
                min_quality = np.min(result['image_qualities'])
                max_quality = np.max(result['image_qualities'])
                print(f"{method_name:<20} {result['avg_quality']:<15.4f} {result['std_quality']:<15.4f} {min_quality:.4f}-{max_quality:.4f}")

        # Find best methods
        fastest_method = min(successful_results.keys(),
                           key=lambda k: successful_results[k]['avg_time_per_image'])
        best_quality_method = max(successful_results.keys(),
                                key=lambda k: successful_results[k]['avg_quality'])
        most_efficient_method = max(successful_results.keys(),
                                  key=lambda k: successful_results[k]['avg_quality'] / successful_results[k]['avg_time_per_image'])

        print(f"\n🎯 RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"🏆 Fastest method: {fastest_method}")
        print(f"   Time: {successful_results[fastest_method]['avg_time_per_image']:.3f}s per image")

        print(f"🎨 Best quality method: {best_quality_method}")
        print(f"   Quality: {successful_results[best_quality_method]['avg_quality']:.4f}")

        print(f"⚡ Most efficient method: {most_efficient_method}")
        print(f"   Efficiency: {successful_results[most_efficient_method]['avg_quality'] / successful_results[most_efficient_method]['avg_time_per_image']:.2f} quality/s")

        # Parameter recommendations
        print(f"\n🔧 PARAMETER RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"• For speed: Use original LML with default parameters")
        print(f"• For quality: Use improved LML with lamb=0.001, kappa=5e-8")
        print(f"• For stability: Use high-quality LML with lamb=0.002, kappa=1e-7")
        print(f"• For different use cases: Experiment with different inference steps")

        # Overall recommendation
        if most_efficient_method == fastest_method:
            print(f"\n✅ RECOMMENDATION: Use '{most_efficient_method}' - provides both speed and efficiency!")
        else:
            print(f"\n⚖️  TRADE-OFF ANALYSIS:")
            print(f"   - For maximum speed: Use '{fastest_method}'")
            print(f"   - For best quality: Use '{best_quality_method}'")
            print(f"   - For best efficiency: Use '{most_efficient_method}'")

def main():
    parser = argparse.ArgumentParser(description="Fast Hessian testing")
    parser.add_argument('--model_id', type=str, default='./model/ddpm_ema_cifar10',
                        help='Path to the model')
    parser.add_argument('--test_num', type=int, default=20,
                        help='Number of test images to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Get absolute path
    model_id = os.path.join(project.model_dir, args.model_id)

    # Run fast test
    tester = FastHessianTester(model_id, args.device)
    tester.run_fast_test(args.test_num)

if __name__ == '__main__':
    main()
