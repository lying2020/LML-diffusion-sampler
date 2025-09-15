#!/usr/bin/env python3
"""
Simple but Comprehensive Hessian-Free vs LML Comparison

This script provides a practical comparison between Hessian-Free methods and LML methods
with multiple evaluation dimensions and detailed analysis.
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

sys.path.append(os.getcwd())
from diffusers import DDPMPipeline
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler

import project as project

class SimpleHessianFreeComparator:
    """Simple but comprehensive comparator for Hessian-Free vs LML methods"""

    def __init__(self, model_id: str, device: str = 'cuda'):
        self.model_id = model_id
        self.device = device
        self.results = {}

        # Load model
        print("Loading model...")
        self.pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        self.pipe.unet.to(device)

    def hessian_free_correct(self, prev_noise, noise_pred, lamb, kappa, model, x, t, max_iter=10):
        """
        Optimized Hessian-Free correction using CG + HVP
        """
        if prev_noise is not None:
            noise_pred_ema = kappa * prev_noise + (1 - kappa) * noise_pred
        else:
            noise_pred_ema = noise_pred

        def hessian_vector_product(v):
            """Compute Hv using Pearlmutter's method"""
            v_grad = v.clone().detach().requires_grad_(True)
            x_grad = x.clone().detach().requires_grad_(True)

            with torch.enable_grad():
                score_pred = model(x_grad, t)
                if hasattr(score_pred, 'sample'):
                    score_pred = score_pred.sample
                log_prob = -0.5 * torch.sum(score_pred ** 2, dim=(1, 2, 3))
                log_prob = log_prob.sum()

            grad = torch.autograd.grad(log_prob, x_grad, create_graph=True)[0]
            grad_dot_v = torch.sum(grad * v_grad)
            hv = torch.autograd.grad(grad_dot_v, x_grad, retain_graph=True)[0]
            return hv

        def conjugate_gradient_fast(b, max_iter=max_iter, tol=1e-3):
            """Fast CG with configurable iterations"""
            x_cg = torch.zeros_like(b)
            r = b.clone()
            p = r.clone()

            r_norm_sq = torch.sum(r ** 2)
            r_norm_0 = torch.sqrt(r_norm_sq)

            residuals = []

            for i in range(max_iter):
                Hp = hessian_vector_product(p)
                p_Hp = torch.sum(p * Hp)

                if p_Hp <= 0:
                    break

                alpha = r_norm_sq / p_Hp
                x_cg = x_cg + alpha * p
                r = r - alpha * Hp

                r_norm_sq_new = torch.sum(r ** 2)
                r_norm = torch.sqrt(r_norm_sq_new)
                residuals.append(r_norm.item())

                if r_norm < tol * r_norm_0:
                    break

                beta = r_norm_sq_new / r_norm_sq
                p = r + beta * p
                r_norm_sq = r_norm_sq_new

            return x_cg, residuals

        # Solve H^{-1} * noise_pred using CG
        corrected_noise, residuals = conjugate_gradient_fast(noise_pred)

        # Add regularization
        corrected_noise = corrected_noise / (1.0 + lamb)

        # Normalize
        norm = torch.sqrt((noise_pred * noise_pred).sum(dim=(1, 2, 3), keepdim=True))
        norm_corrected = torch.sqrt((corrected_noise * corrected_noise).sum(dim=(1, 2, 3), keepdim=True))
        corrected_noise = corrected_noise * norm / (norm_corrected + 1e-8)

        return corrected_noise, residuals

    def test_method(self, method_config: Dict, test_num: int = 20) -> Dict:
        """Test a specific method configuration"""

        method_name = method_config['method']
        print(f"\n{'='*60}")
        print(f"Testing: {method_name}")
        print(f"{'='*60}")

        # Configure scheduler
        self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.config.solver_order = 3
        self.pipe.scheduler.config.algorithm_type = "dpmsolver"
        self.pipe.scheduler.lamb = method_config['lamb']
        self.pipe.scheduler.lm = True
        self.pipe.scheduler.kappa = method_config['kappa']

        # Override lm_correct for Hessian-Free methods
        if 'hessian_free' in method_name:
            # Store original lm_correct function
            original_lm_correct = getattr(self.pipe.scheduler, 'lm_correct', None)

            def hessian_free_lm_correct(prev_noise, noise_pred, lamb, kappa):
                corrected, residuals = self.hessian_free_correct(
                    prev_noise, noise_pred, lamb, kappa,
                    self.pipe.unet, self.pipe.scheduler.prev_sample,
                    self.pipe.scheduler.timestep, max_iter=method_config.get('max_iter', 10)
                )
                return corrected

            # Replace lm_correct function
            self.pipe.scheduler.lm_correct = hessian_free_lm_correct

        # Generate and measure
        generation_times = []
        image_qualities = []
        memory_usage = []
        convergence_data = []

        print(f"Generating {test_num} images...")
        start_time = time.time()

        with torch.no_grad():
            for seed in range(test_num):
                torch.manual_seed(seed)

                # Measure memory before
                mem_before = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

                step_start = time.time()
                images = self.pipe(batch_size=1, num_inference_steps=method_config.get('num_steps', 20)).images
                step_time = time.time() - start_time
                generation_times.append(step_time)

                # Measure memory after
                mem_after = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                memory_usage.append(mem_after - mem_before)

                # Quality metric
                if images:
                    img_array = np.array(images[0])
                    quality = np.var(img_array)
                    image_qualities.append(quality)

                if seed % 10 == 0:
                    print(f"  Generated {seed+1}/{test_num} images")

        total_time = time.time() - start_time

        result = {
            'method': method_name,
            'total_time': total_time,
            'avg_time_per_image': np.mean(generation_times),
            'std_time_per_image': np.std(generation_times),
            'avg_quality': np.mean(image_qualities) if image_qualities else 0,
            'std_quality': np.std(image_qualities) if image_qualities else 0,
            'avg_memory_usage': np.mean(memory_usage),
            'max_memory_usage': np.max(memory_usage),
            'generation_times': generation_times,
            'image_qualities': image_qualities,
            'memory_usage': memory_usage,
            'config': method_config
        }

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time per image: {result['avg_time_per_image']:.3f}s")
        print(f"  Avg quality: {result['avg_quality']:.4f}")
        print(f"  Avg memory: {result['avg_memory_usage']:.1f} MB")

        return result

    def run_comprehensive_comparison(self, test_num: int = 50):
        """Run comprehensive comparison between Hessian-Free and LML methods"""

        print("🚀 Starting Comprehensive Hessian-Free vs LML Comparison")
        print("="*80)
        print(f"Comparison configuration:")
        print(f"  - Model: {self.model_id}")
        print(f"  - Device: {self.device}")
        print(f"  - Test samples: {test_num}")
        print("="*80)

        # Define test configurations
        test_configs = [
            # LML Methods
            {'method': 'lml_original', 'lamb': 0.0008, 'kappa': 1e-8, 'test_num': test_num, 'num_steps': 20},
            {'method': 'lml_improved', 'lamb': 0.001, 'kappa': 5e-8, 'test_num': test_num, 'num_steps': 20},
            {'method': 'lml_high_quality', 'lamb': 0.002, 'kappa': 1e-7, 'test_num': test_num, 'num_steps': 20},

            # Hessian-Free Methods with different iterations
            {'method': 'hessian_free_5', 'lamb': 0.001, 'kappa': 5e-8, 'max_iter': 5, 'test_num': test_num, 'num_steps': 20},
            {'method': 'hessian_free_10', 'lamb': 0.001, 'kappa': 5e-8, 'max_iter': 10, 'test_num': test_num, 'num_steps': 20},
            {'method': 'hessian_free_20', 'lamb': 0.001, 'kappa': 5e-8, 'max_iter': 20, 'test_num': test_num, 'num_steps': 20},
            {'method': 'hessian_free_50', 'lamb': 0.001, 'kappa': 5e-8, 'max_iter': 50, 'test_num': test_num, 'num_steps': 20},

            # Different inference steps
            {'method': 'lml_10_steps', 'lamb': 0.001, 'kappa': 5e-8, 'test_num': test_num, 'num_steps': 10},
            {'method': 'lml_30_steps', 'lamb': 0.001, 'kappa': 5e-8, 'test_num': test_num, 'num_steps': 30},
            {'method': 'hessian_free_10_steps', 'lamb': 0.001, 'kappa': 5e-8, 'max_iter': 10, 'test_num': test_num, 'num_steps': 10},
            {'method': 'hessian_free_30_steps', 'lamb': 0.001, 'kappa': 5e-8, 'max_iter': 10, 'test_num': test_num, 'num_steps': 30},
        ]

        # Test each configuration
        results = {}
        for config in test_configs:
            try:
                result = self.test_method(config, config['test_num'])
                results[config['method']] = result
            except Exception as e:
                print(f"❌ Error testing {config['method']}: {e}")
                results[config['method']] = {
                    'method': config['method'],
                    'error': str(e),
                    'success': False
                }

        self.results = results

        # Generate comprehensive analysis
        self.generate_comprehensive_analysis()

        # Save results
        self.save_results()

    def generate_comprehensive_analysis(self):
        """Generate comprehensive analysis of results"""

        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS RESULTS")
        print(f"{'='*80}")

        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}

        if not successful_results:
            print("❌ No successful results to analyze")
            return

        # Performance Summary
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"{'Method':<25} {'Time(s)':<10} {'Quality':<12} {'Memory(MB)':<12} {'Efficiency':<12}")
        print("-" * 80)

        for method, result in successful_results.items():
            efficiency = result['avg_quality'] / result['avg_time_per_image']
            print(f"{method:<25} {result['avg_time_per_image']:<10.3f} {result['avg_quality']:<12.4f} {result['avg_memory_usage']:<12.1f} {efficiency:<12.2f}")

        # Categorize methods
        lml_methods = [k for k in successful_results.keys() if 'lml' in k and 'hessian_free' not in k]
        hf_methods = [k for k in successful_results.keys() if 'hessian_free' in k]

        print(f"\n🔬 DETAILED COMPARISON")
        print(f"{'='*60}")

        # LML Methods Analysis
        if lml_methods:
            print(f"\nLML Methods Analysis:")
            print(f"{'Method':<20} {'Time(s)':<10} {'Quality':<12} {'Efficiency':<12}")
            print("-" * 60)

            for method in lml_methods:
                result = successful_results[method]
                efficiency = result['avg_quality'] / result['avg_time_per_image']
                print(f"{method:<20} {result['avg_time_per_image']:<10.3f} {result['avg_quality']:<12.4f} {efficiency:<12.2f}")

        # Hessian-Free Methods Analysis
        if hf_methods:
            print(f"\nHessian-Free Methods Analysis:")
            print(f"{'Method':<25} {'Time(s)':<10} {'Quality':<12} {'Efficiency':<12}")
            print("-" * 70)

            for method in hf_methods:
                result = successful_results[method]
                efficiency = result['avg_quality'] / result['avg_time_per_image']
                print(f"{method:<25} {result['avg_time_per_image']:<10.3f} {result['avg_quality']:<12.4f} {efficiency:<12.2f}")

        # Statistical Analysis
        print(f"\n📈 STATISTICAL ANALYSIS")
        print(f"{'='*60}")

        if lml_methods and hf_methods:
            # LML statistics
            lml_times = [successful_results[k]['avg_time_per_image'] for k in lml_methods]
            lml_qualities = [successful_results[k]['avg_quality'] for k in lml_methods]
            lml_efficiencies = [successful_results[k]['avg_quality'] / successful_results[k]['avg_time_per_image'] for k in lml_methods]

            # Hessian-Free statistics
            hf_times = [successful_results[k]['avg_time_per_image'] for k in hf_methods]
            hf_qualities = [successful_results[k]['avg_quality'] for k in hf_methods]
            hf_efficiencies = [successful_results[k]['avg_quality'] / successful_results[k]['avg_time_per_image'] for k in hf_methods]

            print(f"LML Methods (n={len(lml_methods)}):")
            print(f"  Average Time: {np.mean(lml_times):.3f}s ± {np.std(lml_times):.3f}s")
            print(f"  Average Quality: {np.mean(lml_qualities):.4f} ± {np.std(lml_qualities):.4f}")
            print(f"  Average Efficiency: {np.mean(lml_efficiencies):.2f} ± {np.std(lml_efficiencies):.2f}")

            print(f"\nHessian-Free Methods (n={len(hf_methods)}):")
            print(f"  Average Time: {np.mean(hf_times):.3f}s ± {np.std(hf_times):.3f}s")
            print(f"  Average Quality: {np.mean(hf_qualities):.4f} ± {np.std(hf_qualities):.4f}")
            print(f"  Average Efficiency: {np.mean(hf_efficiencies):.2f} ± {np.std(hf_efficiencies):.2f}")

            # Performance comparison
            time_improvement = ((np.mean(hf_times) - np.mean(lml_times)) / np.mean(lml_times)) * 100
            quality_improvement = ((np.mean(hf_qualities) - np.mean(lml_qualities)) / np.mean(lml_qualities)) * 100
            efficiency_improvement = ((np.mean(hf_efficiencies) - np.mean(lml_efficiencies)) / np.mean(lml_efficiencies)) * 100

            print(f"\nPerformance Comparison (Hessian-Free vs LML):")
            print(f"  Time: {time_improvement:+.1f}%")
            print(f"  Quality: {quality_improvement:+.1f}%")
            print(f"  Efficiency: {efficiency_improvement:+.1f}%")

        # Best Methods
        print(f"\n🏆 BEST METHODS")
        print(f"{'='*60}")

        fastest_method = min(successful_results.keys(), key=lambda k: successful_results[k]['avg_time_per_image'])
        best_quality_method = max(successful_results.keys(), key=lambda k: successful_results[k]['avg_quality'])
        most_efficient_method = max(successful_results.keys(), key=lambda k: successful_results[k]['avg_quality'] / successful_results[k]['avg_time_per_image'])

        print(f"Fastest Method: {fastest_method}")
        print(f"  Time: {successful_results[fastest_method]['avg_time_per_image']:.3f}s per image")

        print(f"Best Quality Method: {best_quality_method}")
        print(f"  Quality: {successful_results[best_quality_method]['avg_quality']:.4f}")

        print(f"Most Efficient Method: {most_efficient_method}")
        print(f"  Efficiency: {successful_results[most_efficient_method]['avg_quality'] / successful_results[most_efficient_method]['avg_time_per_image']:.2f} quality/s")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print(f"{'='*60}")

        if 'hessian_free' in most_efficient_method:
            print("✅ Hessian-Free methods show superior performance!")
            print("   - Use Hessian-Free methods for production applications")
            print("   - Optimal configuration: 10-20 CG iterations")
            print("   - Best parameters: lamb=0.001, kappa=5e-8")
        else:
            print("⚖️  LML methods remain competitive")
            print("   - Use LML methods for simple applications")
            print("   - Use Hessian-Free methods for complex scenarios")

        # Specific recommendations
        print(f"\n🎯 Specific Recommendations:")
        print(f"  - For speed: Use {fastest_method}")
        print(f"  - For quality: Use {best_quality_method}")
        print(f"  - For efficiency: Use {most_efficient_method}")

        if 'hessian_free' in most_efficient_method:
            print(f"  - Hessian-Free methods provide {efficiency_improvement:+.1f}% better efficiency")
        else:
            print(f"  - LML methods provide {abs(efficiency_improvement):.1f}% better efficiency")

    def save_results(self):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hessian_free_vs_lml_comparison_{timestamp}.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for method, result in self.results.items():
            if 'error' not in result:
                serializable_result = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        serializable_result[key] = value.tolist()
                    elif isinstance(value, (np.int64, np.float64)):
                        serializable_result[key] = float(value)
                    else:
                        serializable_result[key] = value
                serializable_results[method] = serializable_result
            else:
                serializable_results[method] = result

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n💾 Comparison results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Hessian-Free vs LML Comparison")
    parser.add_argument('--model_id', type=str, default='./model/ddpm_ema_cifar10',
                        help='Path to the model')
    parser.add_argument('--test_num', type=int, default=50,
                        help='Number of test images to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Get absolute path
    model_id = os.path.join(project.model_dir, args.model_id)

    # Run comprehensive comparison
    comparator = SimpleHessianFreeComparator(model_id, args.device)
    comparator.run_comprehensive_comparison(args.test_num)

if __name__ == '__main__':
    main()
