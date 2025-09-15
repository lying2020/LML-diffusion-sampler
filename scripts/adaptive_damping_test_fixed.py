#!/usr/bin/env python3
"""
Adaptive Damping Function Testing (Fixed)

This script tests different adaptive damping functions for Hessian-Free methods
with proper numerical stability handling.
"""

import sys
import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from datetime import datetime
import math

sys.path.append(os.getcwd())
from diffusers import DDPMPipeline
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler

project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'test')

class AdaptiveDampingTester:
    """Tester for different adaptive damping functions with numerical stability"""

    def __init__(self, model_id: str, device: str = 'cuda'):
        self.model_id = model_id
        self.device = device
        self.results = {}

        # Load model
        print("Loading model...")
        self.pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        self.pipe.unet.to(device)

    def adaptive_damping_function(self, condition_number: float, method: str = 'exponential',
                                 lambda_0: float = 0.001, alpha: float = 0.1,
                                 beta: float = 0.5, gamma: float = 0.1, p: float = 2.0) -> float:
        """
        Different adaptive damping functions based on condition number with numerical stability
        """
        # Clamp condition number to prevent numerical issues
        condition_number = max(1.0, min(condition_number, 1e12))

        try:
            if method == 'exponential':
                # λ(κ) = λ₀ * exp(-α * κ)
                exp_arg = -alpha * condition_number
                exp_arg = max(-50, min(exp_arg, 50))  # Prevent overflow
                return lambda_0 * math.exp(exp_arg)

            elif method == 'logarithmic':
                # λ(κ) = λ₀ / (1 + β * log(κ))
                log_val = math.log(max(condition_number, 1.0))
                return lambda_0 / (1 + beta * log_val)

            elif method == 'polynomial':
                # λ(κ) = λ₀ / (1 + γ * κ^p)
                kappa_p = condition_number ** min(p, 10)  # Prevent overflow
                return lambda_0 / (1 + gamma * kappa_p)

            elif method == 'inverse':
                # λ(κ) = λ₀ / κ
                return lambda_0 / max(condition_number, 1.0)

            elif method == 'sigmoid':
                # λ(κ) = λ₀ / (1 + exp(α * (κ - β)))
                exp_arg = alpha * (condition_number - beta)
                exp_arg = max(-50, min(exp_arg, 50))  # Prevent overflow
                return lambda_0 / (1 + math.exp(exp_arg))

            elif method == 'piecewise':
                # Piecewise linear function
                if condition_number < 10:
                    return lambda_0 * 0.1
                elif condition_number < 100:
                    return lambda_0 * 0.5
                elif condition_number < 1000:
                    return lambda_0 * 1.0
                else:
                    return lambda_0 * 2.0

            elif method == 'adaptive':
                # Adaptive based on condition number ranges
                if condition_number < 10:
                    return 0.0001  # Low damping for well-conditioned matrices
                elif condition_number < 100:
                    return 0.001   # Medium damping
                elif condition_number < 1000:
                    return 0.01    # High damping
                else:
                    return 0.1     # Very high damping for ill-conditioned matrices

            else:
                return lambda_0  # Default value

        except (OverflowError, ValueError, ZeroDivisionError):
            return lambda_0  # Fallback to default

    def test_damping_functions(self, condition_numbers: List[float]) -> Dict:
        """Test different damping functions across condition numbers"""

        damping_methods = ['exponential', 'logarithmic', 'polynomial', 'inverse',
                          'sigmoid', 'piecewise', 'adaptive']

        results = {}

        for method in damping_methods:
            damping_values = []
            for cond_num in condition_numbers:
                damping = self.adaptive_damping_function(cond_num, method=method)
                damping_values.append(damping)

            results[method] = {
                'condition_numbers': condition_numbers,
                'damping_values': damping_values,
                'avg_damping': np.mean(damping_values),
                'std_damping': np.std(damping_values),
                'min_damping': np.min(damping_values),
                'max_damping': np.max(damping_values)
            }

        return results

    def plot_damping_functions(self, results: Dict, save_path: str = None):
        """Plot damping functions for visualization"""

        plt.figure(figsize=(15, 10))

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Adaptive Damping Functions Comparison', fontsize=16, fontweight='bold')

        # Plot 1: All functions
        ax1.set_title('All Damping Functions')
        ax1.set_xlabel('Condition Number')
        ax1.set_ylabel('Damping Value')
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        for i, (method, data) in enumerate(results.items()):
            ax1.plot(data['condition_numbers'], data['damping_values'],
                    color=colors[i], label=method, linewidth=2, marker='o', markersize=4)

        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Exponential vs Logarithmic
        ax2.set_title('Exponential vs Logarithmic')
        ax2.set_xlabel('Condition Number')
        ax2.set_ylabel('Damping Value')
        ax2.set_xscale('log')
        ax2.set_yscale('log')

        if 'exponential' in results:
            ax2.plot(results['exponential']['condition_numbers'],
                    results['exponential']['damping_values'],
                    color='red', label='Exponential', linewidth=2, marker='o')
        if 'logarithmic' in results:
            ax2.plot(results['logarithmic']['condition_numbers'],
                    results['logarithmic']['damping_values'],
                    color='blue', label='Logarithmic', linewidth=2, marker='s')

        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Polynomial vs Inverse
        ax3.set_title('Polynomial vs Inverse')
        ax3.set_xlabel('Condition Number')
        ax3.set_ylabel('Damping Value')
        ax3.set_xscale('log')
        ax3.set_yscale('log')

        if 'polynomial' in results:
            ax3.plot(results['polynomial']['condition_numbers'],
                    results['polynomial']['damping_values'],
                    color='green', label='Polynomial', linewidth=2, marker='^')
        if 'inverse' in results:
            ax3.plot(results['inverse']['condition_numbers'],
                    results['inverse']['damping_values'],
                    color='orange', label='Inverse', linewidth=2, marker='d')

        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Adaptive vs Piecewise
        ax4.set_title('Adaptive vs Piecewise')
        ax4.set_xlabel('Condition Number')
        ax4.set_ylabel('Damping Value')
        ax4.set_xscale('log')
        ax4.set_yscale('log')

        if 'adaptive' in results:
            ax4.plot(results['adaptive']['condition_numbers'],
                    results['adaptive']['damping_values'],
                    color='purple', label='Adaptive', linewidth=2, marker='v')
        if 'piecewise' in results:
            ax4.plot(results['piecewise']['condition_numbers'],
                    results['piecewise']['damping_values'],
                    color='brown', label='Piecewise', linewidth=2, marker='<')

        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Damping functions plot saved to: {save_path}")
        else:
            plt.show()

    def test_adaptive_hessian_free_with_damping(self, test_num: int = 20, damping_method: str = 'adaptive') -> Dict:
        """Test Hessian-Free with specific damping method"""

        print(f"\n{'='*60}")
        print(f"Testing Hessian-Free with {damping_method} damping")
        print(f"{'='*60}")

        # Configure scheduler
        self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.config.solver_order = 3
        self.pipe.scheduler.config.algorithm_type = "dpmsolver"
        self.pipe.scheduler.lamb = 0.001
        self.pipe.scheduler.lm = True
        self.pipe.scheduler.kappa = 5e-8

        # Custom Hessian-Free with adaptive damping
        def adaptive_hessian_free_correct(prev_noise, noise_pred, lamb, kappa):
            if prev_noise is not None:
                noise_pred_ema = kappa * prev_noise + (1 - kappa) * noise_pred
            else:
                noise_pred_ema = noise_pred

            def hessian_vector_product(v):
                v_grad = v.clone().detach().requires_grad_(True)
                x_grad = self.pipe.scheduler.prev_sample.clone().detach().requires_grad_(True)

                with torch.enable_grad():
                    score_pred = self.pipe.unet(x_grad, self.pipe.scheduler.timestep)
                    if hasattr(score_pred, 'sample'):
                        score_pred = score_pred.sample
                    log_prob = -0.5 * torch.sum(score_pred ** 2, dim=(1, 2, 3))
                    log_prob = log_prob.sum()

                grad = torch.autograd.grad(log_prob, x_grad, create_graph=True)[0]
                grad_dot_v = torch.sum(grad * v_grad)
                hv = torch.autograd.grad(grad_dot_v, x_grad, retain_graph=True)[0]
                return hv

            def adaptive_conjugate_gradient(b, max_iter=10, tol=1e-3):
                x_cg = torch.zeros_like(b)
                r = b.clone()
                p = r.clone()

                r_norm_sq = torch.sum(r ** 2)
                r_norm_0 = torch.sqrt(r_norm_sq)

                residuals = []
                condition_numbers = []
                adaptive_lambdas = []

                for i in range(max_iter):
                    Hp = hessian_vector_product(p)
                    p_Hp = torch.sum(p * Hp)

                    if p_Hp <= 0:
                        break

                    # Compute condition number approximation
                    if i > 0 and len(residuals) > 1:
                        cond_approx = residuals[0] / (residuals[-1] + 1e-8)
                        condition_numbers.append(cond_approx)

                        # Apply adaptive damping
                        adaptive_lambda = self.adaptive_damping_function(cond_approx, method=damping_method)
                        adaptive_lambdas.append(adaptive_lambda)

                        # Apply adaptive regularization
                        Hp = Hp + adaptive_lambda * p
                        p_Hp = torch.sum(p * Hp)

                    # CG step
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

                return x_cg, residuals, condition_numbers, adaptive_lambdas

            # Solve using adaptive CG
            corrected_noise, residuals, condition_numbers, adaptive_lambdas = adaptive_conjugate_gradient(noise_pred)

            # Apply adaptive regularization
            if adaptive_lambdas:
                avg_lambda = np.mean(adaptive_lambdas)
                corrected_noise = corrected_noise / (1.0 + avg_lambda)
            else:
                corrected_noise = corrected_noise / (1.0 + lamb)

            # Normalize
            norm = torch.sqrt((noise_pred * noise_pred).sum(dim=(1, 2, 3), keepdim=True))
            norm_corrected = torch.sqrt((corrected_noise * corrected_noise).sum(dim=(1, 2, 3), keepdim=True))
            corrected_noise = corrected_noise * norm / (norm_corrected + 1e-8)

            return corrected_noise

        # Override lm_correct
        original_lm_correct = getattr(self.pipe.scheduler, 'lm_correct', None)
        self.pipe.scheduler.lm_correct = adaptive_hessian_free_correct

        # Generate and measure
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

                # Quality metric
                if images:
                    img_array = np.array(images[0])
                    quality = np.var(img_array)
                    image_qualities.append(quality)

                if seed % 5 == 0:
                    print(f"  Generated {seed+1}/{test_num} images")

        total_time = time.time() - start_time

        # Restore original function
        if original_lm_correct:
            self.pipe.scheduler.lm_correct = original_lm_correct

        result = {
            'method': f'hessian_free_{damping_method}',
            'total_time': total_time,
            'avg_time_per_image': np.mean(generation_times),
            'std_time_per_image': np.std(generation_times),
            'avg_quality': np.mean(image_qualities) if image_qualities else 0,
            'std_quality': np.std(image_qualities) if image_qualities else 0,
            'generation_times': generation_times,
            'image_qualities': image_qualities,
            'damping_method': damping_method
        }

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time per image: {result['avg_time_per_image']:.3f}s")
        print(f"  Avg quality: {result['avg_quality']:.4f}")

        return result

    def run_comprehensive_damping_test(self, test_num: int = 20):
        """Run comprehensive test of different damping functions"""

        print("🚀 Starting Comprehensive Adaptive Damping Test")
        print("="*80)

        # Test damping functions across condition numbers
        condition_numbers = [1, 10, 50, 100, 500, 1000, 5000, 10000]
        damping_results = self.test_damping_functions(condition_numbers)

        # Plot damping functions
        self.plot_damping_functions(damping_results, 'adaptive_damping_functions.png')

        # Test each damping method
        damping_methods = ['exponential', 'logarithmic', 'polynomial', 'adaptive']
        test_results = {}

        for method in damping_methods:
            try:
                result = self.test_adaptive_hessian_free_with_damping(test_num, method)
                test_results[method] = result
            except Exception as e:
                print(f"❌ Error testing {method}: {e}")
                test_results[method] = {'error': str(e)}

        # Generate analysis
        self.generate_damping_analysis(damping_results, test_results)

        # Save results
        self.save_damping_results(damping_results, test_results)

    def generate_damping_analysis(self, damping_results: Dict, test_results: Dict):
        """Generate analysis of damping function results"""

        print(f"\n{'='*80}")
        print("ADAPTIVE DAMPING ANALYSIS")
        print(f"{'='*80}")

        # Damping function characteristics
        print(f"\n📊 DAMPING FUNCTION CHARACTERISTICS")
        print(f"{'='*60}")
        print(f"{'Method':<15} {'Avg Damping':<15} {'Std Damping':<15} {'Range':<20}")
        print("-" * 70)

        for method, data in damping_results.items():
            range_str = f"{data['min_damping']:.2e} - {data['max_damping']:.2e}"
            print(f"{method:<15} {data['avg_damping']:<15.2e} {data['std_damping']:<15.2e} {range_str:<20}")

        # Performance comparison
        print(f"\n📈 PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        print(f"{'Method':<15} {'Time(s)':<10} {'Quality':<12} {'Efficiency':<12}")
        print("-" * 50)

        for method, result in test_results.items():
            if 'error' not in result:
                efficiency = result['avg_quality'] / result['avg_time_per_image']
                print(f"{method:<15} {result['avg_time_per_image']:<10.3f} {result['avg_quality']:<12.4f} {efficiency:<12.2f}")

        # Best damping method
        successful_results = {k: v for k, v in test_results.items() if 'error' not in v}
        if successful_results:
            best_method = max(successful_results.keys(),
                            key=lambda k: successful_results[k]['avg_quality'] / successful_results[k]['avg_time_per_image'])
            print(f"\n🏆 Best Damping Method: {best_method}")
            print(f"  Efficiency: {successful_results[best_method]['avg_quality'] / successful_results[best_method]['avg_time_per_image']:.2f} quality/s")

    def save_damping_results(self, damping_results: Dict, test_results: Dict):
        """Save damping test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(project_dir, f"adaptive_damping_test_{timestamp}.json")

        results = {
            'damping_functions': damping_results,
            'performance_tests': test_results,
            'timestamp': timestamp
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Damping test results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Adaptive Damping Function Test (Fixed)")
    parser.add_argument('--model_id', type=str, default='./model/ddpm_ema_cifar10',
                        help='Path to the model')
    parser.add_argument('--test_num', type=int, default=20,
                        help='Number of test images to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Get absolute path
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_id = os.path.join(project_dir, args.model_id)

    # Run damping test
    tester = AdaptiveDampingTester(model_id, args.device)
    tester.run_comprehensive_damping_test(args.test_num)

if __name__ == '__main__':
    main()
