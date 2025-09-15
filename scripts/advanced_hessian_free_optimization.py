#!/usr/bin/env python3
"""
Advanced Hessian-Free Optimization with Adaptive Damping and Condition Number Constraints

This script implements advanced optimizations for Hessian-Free methods including:
1. Adaptive damping based on condition number
2. Condition number constraints
3. Adaptive friction Newton-Langevin
4. Dynamic regularization
5. Convergence monitoring and adaptation
"""

import sys
import os
import time
import torch
import torch.nn as nn
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import math

sys.path.append(os.getcwd())
from diffusers import DDPMPipeline
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
import project as project

class AdvancedHessianFreeOptimizer:
    """Advanced Hessian-Free optimizer with adaptive damping and condition number constraints"""

    def __init__(self, model_id: str, device: str = 'cuda'):
        self.model_id = model_id
        self.device = device
        self.results = {}

        # Load model
        print("Loading model...")
        self.pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        self.pipe.unet.to(device)

        # Adaptive parameters
        self.condition_number_history = []
        self.convergence_history = []
        self.adaptive_lambda_history = []

    def compute_condition_number(self, H_approx: torch.Tensor) -> float:
        """Compute condition number of Hessian approximation"""
        try:
            # Use SVD to compute condition number
            U, S, V = torch.svd(H_approx)
            # Avoid division by zero
            S = S[S > 1e-8]
            if len(S) > 0:
                cond_num = S.max() / S.min()
                return float(cond_num)
            else:
                return 1.0
        except:
            return 1.0

    def adaptive_damping_function(self, condition_number: float, method: str = 'exponential') -> float:
        """
        Adaptive damping function based on condition number

        Args:
            condition_number: Condition number of Hessian
            method: Damping function type ('exponential', 'logarithmic', 'polynomial')
        """
        if method == 'exponential':
            # Exponential damping: λ(κ) = λ₀ * exp(-α * κ)
            lambda_0 = 0.001
            alpha = 0.1
            return lambda_0 * math.exp(-alpha * condition_number)

        elif method == 'logarithmic':
            # Logarithmic damping: λ(κ) = λ₀ / (1 + β * log(κ))
            lambda_0 = 0.001
            beta = 0.5
            return lambda_0 / (1 + beta * math.log(max(condition_number, 1.0)))

        elif method == 'polynomial':
            # Polynomial damping: λ(κ) = λ₀ / (1 + γ * κ^p)
            lambda_0 = 0.001
            gamma = 0.1
            p = 2.0
            return lambda_0 / (1 + gamma * (condition_number ** p))

        elif method == 'adaptive':
            # Adaptive damping based on condition number ranges
            if condition_number < 10:
                return 0.0001  # Low damping for well-conditioned matrices
            elif condition_number < 100:
                return 0.001   # Medium damping
            elif condition_number < 1000:
                return 0.01    # High damping
            else:
                return 0.1     # Very high damping for ill-conditioned matrices

        else:
            return 0.001  # Default value

    def adaptive_friction_newton_langevin_correct(self, prev_noise, noise_pred, lamb, kappa, model, x, t,
                                                max_iter=10, adaptive_damping=True, condition_constraint=True):
        """
        Advanced Hessian-Free correction with adaptive friction Newton-Langevin

        Implements the adaptive version:
        dxt = (1/κ(H_t)) * (H_t + λ(κ(H_t))I)^(-1) * ∇log p(xt)dt + √2M_t dBt
        """
        if prev_noise is not None:
            noise_pred_ema = kappa * prev_noise + (1 - kappa) * noise_pred
        else:
            noise_pred_ema = noise_pred

        def hessian_vector_product_adaptive(v):
            """Compute Hv with adaptive regularization"""
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

        def adaptive_conjugate_gradient(b, max_iter=max_iter, tol=1e-3):
            """Adaptive CG with condition number monitoring and dynamic regularization"""
            x_cg = torch.zeros_like(b)
            r = b.clone()
            p = r.clone()

            r_norm_sq = torch.sum(r ** 2)
            r_norm_0 = torch.sqrt(r_norm_sq)

            residuals = []
            condition_numbers = []
            adaptive_lambdas = []

            for i in range(max_iter):
                # Compute Hp
                Hp = hessian_vector_product_adaptive(p)
                p_Hp = torch.sum(p * Hp)

                # Check for negative curvature
                if p_Hp <= 0:
                    print(f"  Warning: Negative curvature detected at iteration {i}")
                    break

                # Compute condition number approximation
                if i > 0 and len(residuals) > 1:
                    # Approximate condition number using residual ratios
                    cond_approx = residuals[0] / (residuals[-1] + 1e-8)
                    condition_numbers.append(cond_approx)

                    # Adaptive damping based on condition number
                    if adaptive_damping:
                        adaptive_lambda = self.adaptive_damping_function(cond_approx, method='adaptive')
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

                # Check convergence
                if r_norm < tol * r_norm_0:
                    print(f"  CG converged at iteration {i+1}")
                    break

                # Update for next iteration
                beta = r_norm_sq_new / r_norm_sq
                p = r + beta * p
                r_norm_sq = r_norm_sq_new

                # Condition number constraint
                if condition_constraint and len(condition_numbers) > 0:
                    current_cond = condition_numbers[-1]
                    if current_cond > 1e6:  # Threshold for ill-conditioned matrices
                        print(f"  Warning: High condition number {current_cond:.2e}, applying constraint")
                        # Apply additional regularization
                        x_cg = x_cg * 0.9  # Damping factor

            # Store convergence data
            self.convergence_history.append({
                'iterations': i + 1,
                'final_residual': residuals[-1] if residuals else 0,
                'condition_numbers': condition_numbers,
                'adaptive_lambdas': adaptive_lambdas
            })

            return x_cg, residuals, condition_numbers, adaptive_lambdas

        # Solve H^{-1} * noise_pred using adaptive CG
        corrected_noise, residuals, condition_numbers, adaptive_lambdas = adaptive_conjugate_gradient(noise_pred)

        # Store condition number history
        if condition_numbers:
            self.condition_number_history.extend(condition_numbers)
            self.adaptive_lambda_history.extend(adaptive_lambdas)

        # Apply adaptive regularization
        if adaptive_damping and adaptive_lambdas:
            avg_lambda = np.mean(adaptive_lambdas)
            corrected_noise = corrected_noise / (1.0 + avg_lambda)
        else:
            corrected_noise = corrected_noise / (1.0 + lamb)

        # Normalize
        norm = torch.sqrt((noise_pred * noise_pred).sum(dim=(1, 2, 3), keepdim=True))
        norm_corrected = torch.sqrt((corrected_noise * corrected_noise).sum(dim=(1, 2, 3), keepdim=True))
        corrected_noise = corrected_noise * norm / (norm_corrected + 1e-8)

        return corrected_noise, residuals, condition_numbers, adaptive_lambdas

    def test_advanced_hessian_free(self, test_num: int = 20, method_config: Dict = None) -> Dict:
        """Test advanced Hessian-Free method with various optimizations"""

        if method_config is None:
            method_config = {
                'method': 'advanced_hessian_free',
                'lamb': 0.001,
                'kappa': 5e-8,
                'max_iter': 10,
                'adaptive_damping': True,
                'condition_constraint': True,
                'num_steps': 20
            }

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

        # Override lm_correct for advanced Hessian-Free
        original_lm_correct = getattr(self.pipe.scheduler, 'lm_correct', None)

        def advanced_hessian_free_lm_correct(prev_noise, noise_pred, lamb, kappa):
            corrected, residuals, condition_numbers, adaptive_lambdas = self.adaptive_friction_newton_langevin_correct(
                prev_noise, noise_pred, lamb, kappa,
                self.pipe.unet, self.pipe.scheduler.prev_sample,
                self.pipe.scheduler.timestep,
                max_iter=method_config.get('max_iter', 10),
                adaptive_damping=method_config.get('adaptive_damping', True),
                condition_constraint=method_config.get('condition_constraint', True)
            )
            return corrected

        self.pipe.scheduler.lm_correct = advanced_hessian_free_lm_correct

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
                step_time = time.time() - step_start
                generation_times.append(step_time)

                # Measure memory after
                mem_after = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                memory_usage.append(mem_after - mem_before)

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
            'config': method_config,
            'convergence_data': self.convergence_history[-test_num:] if self.convergence_history else [],
            'condition_number_stats': {
                'avg_condition_number': np.mean(self.condition_number_history) if self.condition_number_history else 0,
                'max_condition_number': np.max(self.condition_number_history) if self.condition_number_history else 0,
                'min_condition_number': np.min(self.condition_number_history) if self.condition_number_history else 0,
                'std_condition_number': np.std(self.condition_number_history) if self.condition_number_history else 0
            },
            'adaptive_lambda_stats': {
                'avg_lambda': np.mean(self.adaptive_lambda_history) if self.adaptive_lambda_history else 0,
                'max_lambda': np.max(self.adaptive_lambda_history) if self.adaptive_lambda_history else 0,
                'min_lambda': np.min(self.adaptive_lambda_history) if self.adaptive_lambda_history else 0,
                'std_lambda': np.std(self.adaptive_lambda_history) if self.adaptive_lambda_history else 0
            }
        }

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time per image: {result['avg_time_per_image']:.3f}s")
        print(f"  Avg quality: {result['avg_quality']:.4f}")
        print(f"  Avg memory: {result['avg_memory_usage']:.1f} MB")
        print(f"  Avg condition number: {result['condition_number_stats']['avg_condition_number']:.2e}")
        print(f"  Avg adaptive lambda: {result['adaptive_lambda_stats']['avg_lambda']:.6f}")

        return result

    def run_advanced_optimization_test(self, test_num: int = 30):
        """Run comprehensive test of advanced Hessian-Free optimizations"""

        print("🚀 Starting Advanced Hessian-Free Optimization Test")
        print("="*80)
        print(f"Test configuration:")
        print(f"  - Model: {self.model_id}")
        print(f"  - Device: {self.device}")
        print(f"  - Test samples: {test_num}")
        print("="*80)

        # Define advanced test configurations
        advanced_configs = [
            # Basic Hessian-Free (baseline)
            {
                'method': 'hessian_free_basic',
                'lamb': 0.001,
                'kappa': 5e-8,
                'max_iter': 10,
                'adaptive_damping': False,
                'condition_constraint': False,
                'num_steps': 20
            },

            # Adaptive damping only
            {
                'method': 'hessian_free_adaptive_damping',
                'lamb': 0.001,
                'kappa': 5e-8,
                'max_iter': 10,
                'adaptive_damping': True,
                'condition_constraint': False,
                'num_steps': 20
            },

            # Condition number constraint only
            {
                'method': 'hessian_free_condition_constraint',
                'lamb': 0.001,
                'kappa': 5e-8,
                'max_iter': 10,
                'adaptive_damping': False,
                'condition_constraint': True,
                'num_steps': 20
            },

            # Both adaptive damping and condition constraint
            {
                'method': 'hessian_free_adaptive_friction',
                'lamb': 0.001,
                'kappa': 5e-8,
                'max_iter': 10,
                'adaptive_damping': True,
                'condition_constraint': True,
                'num_steps': 20
            },

            # High iteration count with adaptive features
            {
                'method': 'hessian_free_high_iter',
                'lamb': 0.001,
                'kappa': 5e-8,
                'max_iter': 20,
                'adaptive_damping': True,
                'condition_constraint': True,
                'num_steps': 20
            },

            # Different damping functions
            {
                'method': 'hessian_free_exponential_damping',
                'lamb': 0.001,
                'kappa': 5e-8,
                'max_iter': 10,
                'adaptive_damping': True,
                'condition_constraint': True,
                'damping_method': 'exponential',
                'num_steps': 20
            },

            # Fast generation with optimizations
            {
                'method': 'hessian_free_fast_optimized',
                'lamb': 0.001,
                'kappa': 5e-8,
                'max_iter': 5,
                'adaptive_damping': True,
                'condition_constraint': True,
                'num_steps': 10
            }
        ]

        # Test each configuration
        results = {}
        for config in advanced_configs:
            try:
                # Reset history for each test
                self.condition_number_history = []
                self.convergence_history = []
                self.adaptive_lambda_history = []

                result = self.test_advanced_hessian_free(test_num, config)
                results[config['method']] = result
            except Exception as e:
                print(f"❌ Error testing {config['method']}: {e}")
                results[config['method']] = {
                    'method': config['method'],
                    'error': str(e),
                    'success': False
                }

        self.results = results

        # Generate advanced analysis
        self.generate_advanced_analysis()

        # Save results
        self.save_advanced_results()

    def generate_advanced_analysis(self):
        """Generate advanced analysis of optimization results"""

        print(f"\n{'='*80}")
        print("ADVANCED HESSIAN-FREE OPTIMIZATION ANALYSIS")
        print(f"{'='*80}")

        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}

        if not successful_results:
            print("❌ No successful results to analyze")
            return

        # Performance Summary
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"{'Method':<35} {'Time(s)':<10} {'Quality':<12} {'Memory(MB)':<12} {'Efficiency':<12}")
        print("-" * 90)

        for method, result in successful_results.items():
            efficiency = result['avg_quality'] / result['avg_time_per_image']
            print(f"{method:<35} {result['avg_time_per_image']:<10.3f} {result['avg_quality']:<12.4f} {result['avg_memory_usage']:<12.1f} {efficiency:<12.2f}")

        # Optimization Analysis
        print(f"\n🔬 OPTIMIZATION ANALYSIS")
        print(f"{'='*60}")

        # Compare with and without adaptive features
        basic_result = successful_results.get('hessian_free_basic')
        adaptive_result = successful_results.get('hessian_free_adaptive_friction')

        if basic_result and adaptive_result:
            time_improvement = ((adaptive_result['avg_time_per_image'] - basic_result['avg_time_per_image']) / basic_result['avg_time_per_image']) * 100
            quality_improvement = ((adaptive_result['avg_quality'] - basic_result['avg_quality']) / basic_result['avg_quality']) * 100
            efficiency_improvement = ((adaptive_result['avg_quality'] / adaptive_result['avg_time_per_image']) - (basic_result['avg_quality'] / basic_result['avg_time_per_image'])) / (basic_result['avg_quality'] / basic_result['avg_time_per_image']) * 100

            print(f"Adaptive Friction vs Basic Hessian-Free:")
            print(f"  Time improvement: {time_improvement:+.1f}%")
            print(f"  Quality improvement: {quality_improvement:+.1f}%")
            print(f"  Efficiency improvement: {efficiency_improvement:+.1f}%")

        # Condition Number Analysis
        print(f"\n📈 CONDITION NUMBER ANALYSIS")
        print(f"{'='*60}")

        for method, result in successful_results.items():
            if 'condition_number_stats' in result:
                cond_stats = result['condition_number_stats']
                print(f"{method}:")
                print(f"  Avg condition number: {cond_stats['avg_condition_number']:.2e}")
                print(f"  Max condition number: {cond_stats['max_condition_number']:.2e}")
                print(f"  Min condition number: {cond_stats['min_condition_number']:.2e}")
                print(f"  Std condition number: {cond_stats['std_condition_number']:.2e}")

        # Adaptive Lambda Analysis
        print(f"\n⚙️ ADAPTIVE LAMBDA ANALYSIS")
        print(f"{'='*60}")

        for method, result in successful_results.items():
            if 'adaptive_lambda_stats' in result:
                lambda_stats = result['adaptive_lambda_stats']
                print(f"{method}:")
                print(f"  Avg lambda: {lambda_stats['avg_lambda']:.6f}")
                print(f"  Max lambda: {lambda_stats['max_lambda']:.6f}")
                print(f"  Min lambda: {lambda_stats['min_lambda']:.6f}")
                print(f"  Std lambda: {lambda_stats['std_lambda']:.6f}")

        # Convergence Analysis
        print(f"\n🔄 CONVERGENCE ANALYSIS")
        print(f"{'='*60}")

        for method, result in successful_results.items():
            if 'convergence_data' in result and result['convergence_data']:
                conv_data = result['convergence_data']
                avg_iterations = np.mean([cd['iterations'] for cd in conv_data])
                avg_final_residual = np.mean([cd['final_residual'] for cd in conv_data])
                print(f"{method}:")
                print(f"  Avg iterations: {avg_iterations:.1f}")
                print(f"  Avg final residual: {avg_final_residual:.2e}")

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

        if adaptive_result and basic_result:
            if adaptive_result['avg_time_per_image'] < basic_result['avg_time_per_image']:
                print("✅ Adaptive friction Newton-Langevin shows speed improvement!")
            else:
                print("⚠️  Adaptive features may add computational overhead")

        print("🎯 Optimization recommendations:")
        print("  - Use adaptive damping for ill-conditioned problems")
        print("  - Apply condition number constraints for stability")
        print("  - Monitor convergence for dynamic iteration adjustment")
        print("  - Consider different damping functions for specific use cases")

    def save_advanced_results(self):
        """Save advanced results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(project.output_test_dir, f"advanced_hessian_free_optimization_{timestamp}.json")

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

        print(f"\n💾 Advanced optimization results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Hessian-Free Optimization Test")
    parser.add_argument('--model_id', type=str, default='./model/ddpm_ema_cifar10',
                        help='Path to the model')
    parser.add_argument('--test_num', type=int, default=30,
                        help='Number of test images to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Get absolute path
    model_id = os.path.join(project.model_dir, args.model_id)

    # Run advanced optimization test
    optimizer = AdvancedHessianFreeOptimizer(model_id, args.device)
    optimizer.run_advanced_optimization_test(args.test_num)

if __name__ == '__main__':
    main()
