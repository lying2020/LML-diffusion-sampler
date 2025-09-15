#!/usr/bin/env python3
"""
Comprehensive Hessian Testing Suite for LML Diffusion Sampler

This script provides a complete testing framework for:
1. Original LML method
2. Explicit Hessian computation with finite differences
3. Hessian-Free method using CG + HVP
4. Performance and quality comparison
5. Hessian matrix property analysis
"""

import sys
import os
import time
import torch
import argparse
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime
import project as project

sys.path.append(os.getcwd())
from diffusers import DDPMPipeline
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_dpmsolver_multistep_lm_advanced import DPMSolverMultistepLMSchedulerAdvanced


class ComprehensiveHessianTester:
    """Comprehensive testing suite for Hessian methods"""

    def __init__(self, model_id: str, device: str = 'cuda'):
        self.model_id = model_id
        self.device = device
        self.results = {}

        # Load model
        print("Loading model...")
        self.pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        self.pipe.unet.to(device)

    def test_method(self, method_name: str, method_config: Dict, test_num: int = 10) -> Dict:
        """Test a specific Hessian method"""
        print(f"\n{'='*60}")
        print(f"Testing: {method_config['description']}")
        print(f"{'='*60}")

        # Configure scheduler
        if method_name == 'original':
            self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler.config.solver_order = 3
            self.pipe.scheduler.config.algorithm_type = "dpmsolver"
            self.pipe.scheduler.lamb = method_config['lamb']
            self.pipe.scheduler.lm = True
            self.pipe.scheduler.kappa = method_config['kappa']
        else:
            self.pipe.scheduler = DPMSolverMultistepLMSchedulerAdvanced.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler.config.solver_order = 3
            self.pipe.scheduler.config.algorithm_type = "dpmsolver"
            self.pipe.scheduler.lamb = method_config['lamb']
            self.pipe.scheduler.lm = True
            self.pipe.scheduler.kappa = method_config['kappa']
            self.pipe.scheduler.hessian_method = method_name
            self.pipe.scheduler.set_model(self.pipe.unet)

        # Generate images and measure performance
        generation_times = []
        image_qualities = []
        memory_usage = []

        print(f"Generating {test_num} images...")

        start_time = time.time()

        with torch.no_grad():
            for seed in range(test_num):
                torch.manual_seed(seed)

                # Measure memory before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    memory_before = torch.cuda.memory_allocated()

                step_start = time.time()
                images = self.pipe(batch_size=1, num_inference_steps=20).images
                step_time = time.time() - step_start
                generation_times.append(step_time)

                # Measure memory after generation
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated()
                    memory_usage.append(memory_after - memory_before)

                # Simple quality metric
                if images:
                    img_array = np.array(images[0])
                    # Use variance as quality proxy
                    quality = np.var(img_array)
                    image_qualities.append(quality)

                if seed % 5 == 0:
                    print(f"  Generated {seed+1}/{test_num} images")

        total_time = time.time() - start_time

        # Compute statistics
        result = {
            'method': method_name,
            'description': method_config['description'],
            'total_time': total_time,
            'avg_time_per_image': np.mean(generation_times),
            'std_time_per_image': np.std(generation_times),
            'min_time': np.min(generation_times),
            'max_time': np.max(generation_times),
            'avg_quality': np.mean(image_qualities) if image_qualities else 0,
            'std_quality': np.std(image_qualities) if image_qualities else 0,
            'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0,
            'generation_times': generation_times,
            'image_qualities': image_qualities,
            'memory_usage': memory_usage
        }

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per image: {result['avg_time_per_image']:.3f}s")
        print(f"  Average quality: {result['avg_quality']:.4f}")
        if memory_usage:
            print(f"  Average memory usage: {result['avg_memory_usage']/1024**2:.1f} MB")

        return result

    def run_comprehensive_test(self, test_num: int = 20):
        """Run comprehensive test of all methods"""

        print("🚀 Starting Comprehensive Hessian Testing Suite")
        print("="*60)
        print(f"Test configuration:")
        print(f"  - Model: {self.model_id}")
        print(f"  - Device: {self.device}")
        print(f"  - Test samples: {test_num}")
        print(f"  - Inference steps: 20")
        print("="*60)

        # Define test methods
        methods = {
            'original': {
                'description': 'Original LML (simplified approximation)',
                'lamb': 0.0008,
                'kappa': 1.0e-8
            },
            'explicit': {
                'description': 'Explicit Hessian computation (finite differences)',
                'lamb': 0.0008,
                'kappa': 1.0e-8
            },
            'hessian_free': {
                'description': 'Hessian-Free method (CG + HVP)',
                'lamb': 0.0008,
                'kappa': 1.0e-8
            }
        }

        # Test each method
        for method_name, method_config in methods.items():
            try:
                result = self.test_method(method_name, method_config, test_num)
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

        # Save results
        self.save_results()

    def generate_comparison_report(self):
        """Generate detailed comparison report"""

        print(f"\n{'='*60}")
        print("COMPREHENSIVE COMPARISON REPORT")
        print(f"{'='*60}")

        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}

        if not successful_results:
            print("❌ No successful results to compare")
            return

        # Performance comparison
        print(f"\n📊 PERFORMANCE COMPARISON")
        print(f"{'Method':<20} {'Time (s)':<12} {'Quality':<12} {'Memory (MB)':<12} {'Efficiency':<12}")
        print("-" * 80)

        baseline_time = min(r['avg_time_per_image'] for r in successful_results.values())

        for method_name, result in successful_results.items():
            efficiency = baseline_time / result['avg_time_per_image']
            memory_mb = result['avg_memory_usage'] / 1024**2 if result['avg_memory_usage'] > 0 else 0

            print(f"{method_name:<20} {result['avg_time_per_image']:<12.3f} {result['avg_quality']:<12.4f} {memory_mb:<12.1f} {efficiency:<12.2f}x")

        # Quality comparison
        print(f"\n🎨 QUALITY COMPARISON")
        print(f"{'Method':<20} {'Avg Quality':<15} {'Std Quality':<15} {'Quality Range':<15}")
        print("-" * 70)

        for method_name, result in successful_results.items():
            if result['image_qualities']:
                min_quality = np.min(result['image_qualities'])
                max_quality = np.max(result['image_qualities'])
                print(f"{method_name:<20} {result['avg_quality']:<15.4f} {result['std_quality']:<15.4f} {min_quality:.4f}-{max_quality:.4f}")

        # Stability comparison
        print(f"\n⚡ STABILITY COMPARISON")
        print(f"{'Method':<20} {'Time Std':<15} {'Quality Std':<15} {'Consistency':<15}")
        print("-" * 70)

        for method_name, result in successful_results.items():
            time_cv = result['std_time_per_image'] / result['avg_time_per_image'] if result['avg_time_per_image'] > 0 else 0
            quality_cv = result['std_quality'] / result['avg_quality'] if result['avg_quality'] > 0 else 0
            consistency = 1.0 / (1.0 + time_cv + quality_cv)  # Higher is better

            print(f"{method_name:<20} {result['std_time_per_image']:<15.3f} {result['std_quality']:<15.4f} {consistency:<15.3f}")

        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS")
        print(f"{'='*60}")

        if successful_results:
            # Find best methods
            fastest_method = min(successful_results.keys(),
                               key=lambda k: successful_results[k]['avg_time_per_image'])
            best_quality_method = max(successful_results.keys(),
                                    key=lambda k: successful_results[k]['avg_quality'])
            most_stable_method = min(successful_results.keys(),
                                   key=lambda k: successful_results[k]['std_time_per_image'])

            print(f"🏆 Fastest method: {fastest_method}")
            print(f"   Time: {successful_results[fastest_method]['avg_time_per_image']:.3f}s per image")

            print(f"🎨 Best quality method: {best_quality_method}")
            print(f"   Quality: {successful_results[best_quality_method]['avg_quality']:.4f}")

            print(f"⚡ Most stable method: {most_stable_method}")
            print(f"   Time std: {successful_results[most_stable_method]['std_time_per_image']:.3f}s")

            # Overall recommendation
            if fastest_method == best_quality_method:
                print(f"\n✅ RECOMMENDATION: Use '{fastest_method}' - provides both speed and quality!")
            else:
                print(f"\n⚖️  TRADE-OFF ANALYSIS:")
                print(f"   - For speed: Use '{fastest_method}'")
                print(f"   - For quality: Use '{best_quality_method}'")
                print(f"   - For stability: Use '{most_stable_method}'")

    def save_results(self):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hessian_test_results_{timestamp}.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for method, result in self.results.items():
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, (np.int64, np.float64)):
                    serializable_result[key] = float(value)
                else:
                    serializable_result[key] = value
            serializable_results[method] = serializable_result

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n💾 Results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Hessian testing suite")
    parser.add_argument('--model_id', type=str, default='./model/ddpm_ema_cifar10',
                        help='Path to the model')
    parser.add_argument('--test_num', type=int, default=20,
                        help='Number of test images to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Get absolute path
    model_id = os.path.join(project.model_dir, args.model_id)

    # Run comprehensive test
    tester = ComprehensiveHessianTester(model_id, args.device)
    tester.run_comprehensive_test(args.test_num)

if __name__ == '__main__':
    main()
