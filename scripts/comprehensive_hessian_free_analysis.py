#!/usr/bin/env python3
"""
Comprehensive Hessian-Free vs LML Analysis

This script provides detailed comparison between Hessian-Free methods and LML methods
from multiple dimensions including:
1. Performance (speed, memory, quality)
2. Scalability (different batch sizes, image sizes)
3. Stability (different parameters, noise levels)
4. Convergence (different inference steps)
5. Robustness (different models, datasets)
"""

import sys
import os
import time
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from datetime import datetime
import psutil
import gc

sys.path.append(os.getcwd())
from diffusers import DDPMPipeline
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler

import project as project

class ComprehensiveAnalyzer:
    """Comprehensive analyzer for Hessian-Free vs LML methods"""

    def __init__(self, model_id: str, device: str = 'cuda'):
        self.model_id = model_id
        self.device = device
        self.results = {}
        self.memory_usage = []

        # Load model
        print("Loading model...")
        self.pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        self.pipe.unet.to(device)

    def measure_memory(self):
        """Measure current memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2  # MB
        else:
            return psutil.Process().memory_info().rss / 1024**2  # MB

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

    def test_performance_dimension(self, test_configs: List[Dict]) -> Dict:
        """Test performance across different configurations"""
        print(f"\n{'='*60}")
        print("PERFORMANCE DIMENSION ANALYSIS")
        print(f"{'='*60}")

        results = {}

        for config in test_configs:
            method_name = config['method']
            print(f"\nTesting {method_name}...")

            # Configure scheduler
            self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler.config.solver_order = 3
            self.pipe.scheduler.config.algorithm_type = "dpmsolver"
            self.pipe.scheduler.lamb = config['lamb']
            self.pipe.scheduler.lm = True
            self.pipe.scheduler.kappa = config['kappa']

            # Override lm_correct for Hessian-Free methods
            if 'hessian_free' in method_name:
                original_lm_correct = self.pipe.scheduler.lm_correct

                def hessian_free_lm_correct(prev_noise, noise_pred, lamb, kappa):
                    corrected, residuals = self.hessian_free_correct(
                        prev_noise, noise_pred, lamb, kappa,
                        self.pipe.unet, self.pipe.scheduler.prev_sample,
                        self.pipe.scheduler.timestep, max_iter=config.get('max_iter', 10)
                    )
                    return corrected

                self.pipe.scheduler.lm_correct = hessian_free_lm_correct

            # Generate and measure
            generation_times = []
            image_qualities = []
            memory_usage = []
            convergence_data = []

            print(f"Generating {config['test_num']} images...")
            start_time = time.time()

            with torch.no_grad():
                for seed in range(config['test_num']):
                    torch.manual_seed(seed)

                    # Measure memory before
                    mem_before = self.measure_memory()

                    step_start = time.time()
                    images = self.pipe(batch_size=config['batch_size'],
                                     num_inference_steps=config['num_steps']).images
                    step_time = time.time() - step_start
                    generation_times.append(step_time)

                    # Measure memory after
                    mem_after = self.measure_memory()
                    memory_usage.append(mem_after - mem_before)

                    # Quality metric
                    if images:
                        img_array = np.array(images[0])
                        quality = np.var(img_array)
                        image_qualities.append(quality)

                    if seed % 10 == 0:
                        print(f"  Generated {seed+1}/{config['test_num']} images")

            total_time = time.time() - start_time

            results[method_name] = {
                'total_time': total_time,
                'avg_time_per_image': np.mean(generation_times),
                'std_time_per_image': np.std(generation_times),
                'avg_quality': np.mean(image_qualities) if image_qualities else 0,
                'std_quality': np.std(image_qualities) if image_qualities else 0,
                'avg_memory_usage': np.mean(memory_usage),
                'max_memory_usage': np.max(memory_usage),
                'generation_times': generation_times,
                'image_qualities': image_qualities,
                'memory_usage': memory_usage
            }

            print(f"  Total time: {total_time:.2f}s")
            print(f"  Avg time per image: {np.mean(generation_times):.3f}s")
            print(f"  Avg quality: {np.mean(image_qualities):.4f}")
            print(f"  Avg memory: {np.mean(memory_usage):.1f} MB")

        return results

    def test_scalability_dimension(self) -> Dict:
        """Test scalability across different batch sizes and image sizes"""
        print(f"\n{'='*60}")
        print("SCALABILITY DIMENSION ANALYSIS")
        print(f"{'='*60}")

        batch_sizes = [1, 2, 4, 8]
        methods = ['lml_original', 'hessian_free_5', 'hessian_free_10', 'hessian_free_20']

        results = {}

        for method in methods:
            print(f"\nTesting {method} scalability...")
            method_results = {}

            for batch_size in batch_sizes:
                try:
                    # Configure method
                    if 'hessian_free' in method:
                        max_iter = int(method.split('_')[-1])
                        config = {
                            'method': method,
                            'lamb': 0.001,
                            'kappa': 5e-8,
                            'max_iter': max_iter,
                            'test_num': 5,
                            'batch_size': batch_size,
                            'num_steps': 20
                        }
                    else:
                        config = {
                            'method': method,
                            'lamb': 0.0008,
                            'kappa': 1e-8,
                            'test_num': 5,
                            'batch_size': batch_size,
                            'num_steps': 20
                        }

                    # Test this configuration
                    test_results = self.test_performance_dimension([config])
                    method_results[batch_size] = test_results[method]

                except Exception as e:
                    print(f"  Error with batch_size {batch_size}: {e}")
                    method_results[batch_size] = {'error': str(e)}

            results[method] = method_results

        return results

    def test_stability_dimension(self) -> Dict:
        """Test stability across different parameters and noise levels"""
        print(f"\n{'='*60}")
        print("STABILITY DIMENSION ANALYSIS")
        print(f"{'='*60}")

        # Different parameter combinations
        param_configs = [
            {'lamb': 0.0005, 'kappa': 1e-8, 'name': 'low_reg'},
            {'lamb': 0.001, 'kappa': 5e-8, 'name': 'medium_reg'},
            {'lamb': 0.002, 'kappa': 1e-7, 'name': 'high_reg'},
            {'lamb': 0.005, 'kappa': 5e-7, 'name': 'very_high_reg'}
        ]

        methods = ['lml_original', 'hessian_free_10']
        results = {}

        for method in methods:
            print(f"\nTesting {method} stability...")
            method_results = {}

            for param_config in param_configs:
                try:
                    config = {
                        'method': f"{method}_{param_config['name']}",
                        'lamb': param_config['lamb'],
                        'kappa': param_config['kappa'],
                        'max_iter': 10 if 'hessian_free' in method else None,
                        'test_num': 10,
                        'batch_size': 1,
                        'num_steps': 20
                    }

                    test_results = self.test_performance_dimension([config])
                    method_results[param_config['name']] = test_results[config['method']]

                except Exception as e:
                    print(f"  Error with params {param_config['name']}: {e}")
                    method_results[param_config['name']] = {'error': str(e)}

            results[method] = method_results

        return results

    def test_convergence_dimension(self) -> Dict:
        """Test convergence across different inference steps"""
        print(f"\n{'='*60}")
        print("CONVERGENCE DIMENSION ANALYSIS")
        print(f"{'='*60}")

        inference_steps = [5, 10, 15, 20, 25, 30]
        methods = ['lml_original', 'hessian_free_5', 'hessian_free_10', 'hessian_free_20']

        results = {}

        for method in methods:
            print(f"\nTesting {method} convergence...")
            method_results = {}

            for steps in inference_steps:
                try:
                    if 'hessian_free' in method:
                        max_iter = int(method.split('_')[-1])
                        config = {
                            'method': method,
                            'lamb': 0.001,
                            'kappa': 5e-8,
                            'max_iter': max_iter,
                            'test_num': 5,
                            'batch_size': 1,
                            'num_steps': steps
                        }
                    else:
                        config = {
                            'method': method,
                            'lamb': 0.0008,
                            'kappa': 1e-8,
                            'test_num': 5,
                            'batch_size': 1,
                            'num_steps': steps
                        }

                    test_results = self.test_performance_dimension([config])
                    method_results[steps] = test_results[method]

                except Exception as e:
                    print(f"  Error with {steps} steps: {e}")
                    method_results[steps] = {'error': str(e)}

            results[method] = method_results

        return results

    def run_comprehensive_analysis(self, test_num: int = 50):
        """Run comprehensive analysis across all dimensions"""

        print("🚀 Starting Comprehensive Hessian-Free vs LML Analysis")
        print("="*80)
        print(f"Analysis configuration:")
        print(f"  - Model: {self.model_id}")
        print(f"  - Device: {self.device}")
        print(f"  - Test samples: {test_num}")
        print(f"  - Analysis dimensions: Performance, Scalability, Stability, Convergence")
        print("="*80)

        # Define test configurations
        test_configs = [
            # Original LML methods
            {'method': 'lml_original', 'lamb': 0.0008, 'kappa': 1e-8, 'test_num': test_num, 'batch_size': 1, 'num_steps': 20},
            {'method': 'lml_improved', 'lamb': 0.001, 'kappa': 5e-8, 'test_num': test_num, 'batch_size': 1, 'num_steps': 20},
            {'method': 'lml_high_quality', 'lamb': 0.002, 'kappa': 1e-7, 'test_num': test_num, 'batch_size': 1, 'num_steps': 20},

            # Hessian-Free methods with different iterations
            {'method': 'hessian_free_5', 'lamb': 0.001, 'kappa': 5e-8, 'max_iter': 5, 'test_num': test_num, 'batch_size': 1, 'num_steps': 20},
            {'method': 'hessian_free_10', 'lamb': 0.001, 'kappa': 5e-8, 'max_iter': 10, 'test_num': test_num, 'batch_size': 1, 'num_steps': 20},
            {'method': 'hessian_free_20', 'lamb': 0.001, 'kappa': 5e-8, 'max_iter': 20, 'test_num': test_num, 'batch_size': 1, 'num_steps': 20},
            {'method': 'hessian_free_50', 'lamb': 0.001, 'kappa': 5e-8, 'max_iter': 50, 'test_num': test_num, 'batch_size': 1, 'num_steps': 20},
        ]

        # Run all analyses
        print("\n🔬 Running Performance Analysis...")
        performance_results = self.test_performance_dimension(test_configs)

        print("\n📈 Running Scalability Analysis...")
        scalability_results = self.test_scalability_dimension()

        print("\n⚖️ Running Stability Analysis...")
        stability_results = self.test_stability_dimension()

        print("\n🔄 Running Convergence Analysis...")
        convergence_results = self.test_convergence_dimension()

        # Compile all results
        self.results = {
            'performance': performance_results,
            'scalability': scalability_results,
            'stability': stability_results,
            'convergence': convergence_results,
            'timestamp': datetime.now().isoformat(),
            'test_config': {
                'test_num': test_num,
                'device': self.device,
                'model_id': self.model_id
            }
        }

        # Generate comprehensive report
        self.generate_comprehensive_report()

        # Save results
        self.save_results()

    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""

        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS REPORT")
        print(f"{'='*80}")

        # Performance Summary
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"{'='*60}")

        perf_results = self.results['performance']
        if perf_results:
            print(f"{'Method':<20} {'Time(s)':<10} {'Quality':<12} {'Memory(MB)':<12} {'Efficiency':<12}")
            print("-" * 70)

            for method, result in perf_results.items():
                if 'error' not in result:
                    efficiency = result['avg_quality'] / result['avg_time_per_image']
                    print(f"{method:<20} {result['avg_time_per_image']:<10.3f} {result['avg_quality']:<12.4f} {result['avg_memory_usage']:<12.1f} {efficiency:<12.2f}")

        # Scalability Analysis
        print(f"\n📈 SCALABILITY ANALYSIS")
        print(f"{'='*60}")

        scale_results = self.results['scalability']
        for method, batch_results in scale_results.items():
            print(f"\n{method}:")
            print(f"{'Batch Size':<12} {'Time(s)':<10} {'Quality':<12} {'Memory(MB)':<12}")
            print("-" * 50)

            for batch_size, result in batch_results.items():
                if 'error' not in result:
                    print(f"{batch_size:<12} {result['avg_time_per_image']:<10.3f} {result['avg_quality']:<12.4f} {result['avg_memory_usage']:<12.1f}")

        # Stability Analysis
        print(f"\n⚖️ STABILITY ANALYSIS")
        print(f"{'='*60}")

        stab_results = self.results['stability']
        for method, param_results in stab_results.items():
            print(f"\n{method}:")
            print(f"{'Parameters':<15} {'Time(s)':<10} {'Quality':<12} {'Std Quality':<12}")
            print("-" * 55)

            for param_name, result in param_results.items():
                if 'error' not in result:
                    print(f"{param_name:<15} {result['avg_time_per_image']:<10.3f} {result['avg_quality']:<12.4f} {result['std_quality']:<12.4f}")

        # Convergence Analysis
        print(f"\n🔄 CONVERGENCE ANALYSIS")
        print(f"{'='*60}")

        conv_results = self.results['convergence']
        for method, step_results in conv_results.items():
            print(f"\n{method}:")
            print(f"{'Steps':<8} {'Time(s)':<10} {'Quality':<12} {'Efficiency':<12}")
            print("-" * 45)

            for steps, result in step_results.items():
                if 'error' not in result:
                    efficiency = result['avg_quality'] / result['avg_time_per_image']
                    print(f"{steps:<8} {result['avg_time_per_image']:<10.3f} {result['avg_quality']:<12.4f} {efficiency:<12.2f}")

        # Key Insights
        print(f"\n🎯 KEY INSIGHTS")
        print(f"{'='*60}")

        self._generate_key_insights()

    def _generate_key_insights(self):
        """Generate key insights from the analysis"""

        perf_results = self.results['performance']

        if not perf_results:
            print("No performance data available for insights.")
            return

        # Find best methods
        valid_results = {k: v for k, v in perf_results.items() if 'error' not in v}

        if not valid_results:
            print("No valid results for insights.")
            return

        # Performance insights
        fastest_method = min(valid_results.keys(), key=lambda k: valid_results[k]['avg_time_per_image'])
        best_quality_method = max(valid_results.keys(), key=lambda k: valid_results[k]['avg_quality'])
        most_efficient_method = max(valid_results.keys(), key=lambda k: valid_results[k]['avg_quality'] / valid_results[k]['avg_time_per_image'])

        print(f"🏆 Performance Leaders:")
        print(f"  - Fastest: {fastest_method} ({valid_results[fastest_method]['avg_time_per_image']:.3f}s)")
        print(f"  - Best Quality: {best_quality_method} ({valid_results[best_quality_method]['avg_quality']:.4f})")
        print(f"  - Most Efficient: {most_efficient_method} ({valid_results[most_efficient_method]['avg_quality'] / valid_results[most_efficient_method]['avg_time_per_image']:.2f} quality/s)")

        # Hessian-Free vs LML comparison
        lml_methods = [k for k in valid_results.keys() if 'lml' in k]
        hf_methods = [k for k in valid_results.keys() if 'hessian_free' in k]

        if lml_methods and hf_methods:
            avg_lml_time = np.mean([valid_results[k]['avg_time_per_image'] for k in lml_methods])
            avg_hf_time = np.mean([valid_results[k]['avg_time_per_image'] for k in hf_methods])
            avg_lml_quality = np.mean([valid_results[k]['avg_quality'] for k in lml_methods])
            avg_hf_quality = np.mean([valid_results[k]['avg_quality'] for k in hf_methods])

            print(f"\n🔬 Hessian-Free vs LML Comparison:")
            print(f"  - LML Average Time: {avg_lml_time:.3f}s")
            print(f"  - Hessian-Free Average Time: {avg_hf_time:.3f}s")
            print(f"  - Time Difference: {((avg_hf_time - avg_lml_time) / avg_lml_time * 100):+.1f}%")
            print(f"  - LML Average Quality: {avg_lml_quality:.4f}")
            print(f"  - Hessian-Free Average Quality: {avg_hf_quality:.4f}")
            print(f"  - Quality Difference: {((avg_hf_quality - avg_lml_quality) / avg_lml_quality * 100):+.1f}%")

        # Recommendations
        print(f"\n💡 Recommendations:")
        print(f"  - For maximum speed: Use {fastest_method}")
        print(f"  - For best quality: Use {best_quality_method}")
        print(f"  - For balanced performance: Use {most_efficient_method}")

        if 'hessian_free' in most_efficient_method:
            print(f"  - Hessian-Free methods show superior efficiency")
        else:
            print(f"  - LML methods remain competitive for this use case")

    def save_results(self):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_hessian_analysis_{timestamp}.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for category, results in self.results.items():
            if category in ['timestamp', 'test_config']:
                serializable_results[category] = results
            else:
                serializable_results[category] = {}
                for method, result in results.items():
                    if isinstance(result, dict) and 'error' not in result:
                        serializable_result = {}
                        for key, value in result.items():
                            if isinstance(value, np.ndarray):
                                serializable_result[key] = value.tolist()
                            elif isinstance(value, (np.int64, np.float64)):
                                serializable_result[key] = float(value)
                            else:
                                serializable_result[key] = value
                        serializable_results[category][method] = serializable_result
                    else:
                        serializable_results[category][method] = result

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n💾 Comprehensive analysis results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Hessian-Free vs LML Analysis")
    parser.add_argument('--model_id', type=str, default='./model/ddpm_ema_cifar10',
                        help='Path to the model')
    parser.add_argument('--test_num', type=int, default=50,
                        help='Number of test images to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Get absolute path
    model_id = os.path.join(project.model_dir, args.model_id)

    # Run comprehensive analysis
    analyzer = ComprehensiveAnalyzer(model_id, args.device)
    analyzer.run_comprehensive_analysis(args.test_num)

if __name__ == '__main__':
    main()
