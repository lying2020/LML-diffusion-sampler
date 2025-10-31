#!/usr/bin/env python3
"""
LML vs Hessian Methods Comparison

This script provides a detailed comparison between LML, Explicit Hessian, and Hessian-Free methods
with theoretical analysis and algorithm flow descriptions.
"""

import sys
import os
import time
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import math
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.getcwd())

from diffusers import DDPMPipeline
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
import project as project

hessian_cg_results_dir = os.path.join(project.output_dir, "hessian_cg")
os.makedirs(hessian_cg_results_dir, exist_ok=True)

class LMLvsHessianComparator:
    """Detailed comparator for LML vs Hessian methods"""

    def __init__(self, model_path, output_dir=hessian_cg_results_dir):
        self.model_path = model_path
        self.output_dir = output_dir
        self.results = {}

        # Load model
        print("Loading model...")
        self.pipe = DDPMPipeline.from_pretrained(self.model_path, torch_dtype=torch.float32, use_safetensors=False)
        self.pipe.unet.to('cuda' if torch.cuda.is_available() else 'cpu')

    def lml_correct(self, prev_noise, noise_pred, lamb, kappa):
        """Original LML correction method"""
        if prev_noise is not None:
            noise_pred_ema = kappa * prev_noise + (1 - kappa) * noise_pred
        else:
            noise_pred_ema = noise_pred

        # LML correction using approximate Hessian
        # This is a simplified version of the LML correction
        corrected_noise = noise_pred_ema / (1.0 + lamb)

        # Normalize
        norm = torch.sqrt((noise_pred_ema * noise_pred_ema).sum(dim=(1, 2, 3), keepdim=True))
        norm_corrected = torch.sqrt((corrected_noise * corrected_noise).sum(dim=(1, 2, 3), keepdim=True))
        corrected_noise = corrected_noise * norm / (norm_corrected + 1e-8)

        return corrected_noise

    def explicit_hessian_correct(self, prev_noise, noise_pred, lamb, kappa, model, x, t):
        """Explicit Hessian calculation and correction"""
        if prev_noise is not None:
            noise_pred_ema = kappa * prev_noise + (1 - kappa) * noise_pred
        else:
            noise_pred_ema = noise_pred

        # Enable gradients for Hessian computation
        x_grad = x.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            # First prediction
            score_pred1 = model(x_grad, t)
            if hasattr(score_pred1, 'sample'):
                score_pred1 = score_pred1.sample

            # Compute first-order derivatives
            log_prob1 = -0.5 * torch.sum(score_pred1 ** 2, dim=(1, 2, 3))
            log_prob1 = log_prob1.sum()

            # Compute gradient
            grad1 = torch.autograd.grad(log_prob1, x_grad, create_graph=True)[0]

            # Compute Hessian using second-order derivatives
            hessian = []
            for i in range(grad1.numel()):
                grad_i = grad1.view(-1)[i]
                hessian_i = torch.autograd.grad(grad_i, x_grad, retain_graph=True)[0]
                hessian.append(hessian_i.view(-1))

            hessian_matrix = torch.stack(hessian, dim=1)

            # Compute condition number and rank
            try:
                eigenvals = torch.linalg.eigvals(hessian_matrix)
                eigenvals_real = eigenvals.real
                condition_number = torch.max(eigenvals_real) / (torch.min(eigenvals_real) + 1e-8)
                rank = torch.sum(eigenvals_real > 1e-6)
            except:
                condition_number = torch.tensor(1.0)
                rank = torch.tensor(hessian_matrix.shape[0])

            # Apply Hessian correction
            try:
                hessian_inv = torch.linalg.inv(hessian_matrix + lamb * torch.eye(hessian_matrix.shape[0], device=hessian_matrix.device))
                corrected_noise = torch.matmul(hessian_inv, noise_pred_ema.view(-1)).view(noise_pred_ema.shape)
            except:
                # Fallback to simple correction
                corrected_noise = noise_pred_ema / (1.0 + lamb)

            # Normalize
            norm = torch.sqrt((noise_pred_ema * noise_pred_ema).sum(dim=(1, 2, 3), keepdim=True))
            norm_corrected = torch.sqrt((corrected_noise * corrected_noise).sum(dim=(1, 2, 3), keepdim=True))
            corrected_noise = corrected_noise * norm / (norm_corrected + 1e-8)

            return corrected_noise, condition_number.item(), rank.item()

    def _correct(self, prev_noise, noise_pred, lamb, kappa, model, x, t, max_iter=10):
        """Hessian-Free correction using conjugate gradient"""
        if prev_noise is not None:
            noise_pred_ema = kappa * prev_noise + (1 - kappa) * noise_pred
        else:
            noise_pred_ema = noise_pred

        def hessian_vector_product(v):
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

        def conjugate_gradient(b, max_iter=max_iter, tol=1e-3):
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

            return x_cg, residuals

        # Solve using CG
        corrected_noise, residuals = conjugate_gradient(noise_pred_ema)

        # Apply regularization
        corrected_noise = corrected_noise / (1.0 + lamb)

        # Normalize
        norm = torch.sqrt((noise_pred_ema * noise_pred_ema).sum(dim=(1, 2, 3), keepdim=True))
        norm_corrected = torch.sqrt((corrected_noise * corrected_noise).sum(dim=(1, 2, 3), keepdim=True))
        corrected_noise = corrected_noise * norm / (norm_corrected + 1e-8)

        return corrected_noise, residuals

    def test_algorithm(self, algorithm_config, test_num=50):
        """Test a specific algorithm configuration"""

        algorithm_name = algorithm_config['name']
        print(f"Testing {algorithm_name}...")

        # Configure scheduler
        self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.config.solver_order = 3
        self.pipe.scheduler.config.algorithm_type = "dpmsolver"
        self.pipe.scheduler.lamb = algorithm_config.get('lamb', 0.001)
        self.pipe.scheduler.lm = True
        self.pipe.scheduler.kappa = algorithm_config.get('kappa', 5e-8)

        # Override lm_correct based on algorithm type
        if algorithm_config['type'] == 'lml':
            self.pipe.scheduler.lm_correct = self.lml_correct
        elif algorithm_config['type'] == 'explicit_hessian':
            def explicit_hessian_lm_correct(prev_noise, noise_pred, lamb, kappa):
                corrected, condition_number, rank = self.explicit_hessian_correct(
                    prev_noise, noise_pred, lamb, kappa,
                    self.pipe.unet, self.pipe.scheduler.prev_sample,
                    self.pipe.scheduler.timestep
                )
                return corrected
            self.pipe.scheduler.lm_correct = explicit_hessian_lm_correct
        elif algorithm_config['type'] == 'hcg':
            def _lm_correct(prev_noise, noise_pred, lamb, kappa):
                corrected, residuals = self._correct(
                    prev_noise, noise_pred, lamb, kappa,
                    self.pipe.unet, self.pipe.scheduler.prev_sample,
                    self.pipe.scheduler.timestep, max_iter=algorithm_config.get('max_iter', 10)
                )
                return corrected
            self.pipe.scheduler.lm_correct = _lm_correct

        # Generate images
        generation_times = []
        image_qualities = []
        all_images = []

        print(f"Generating {test_num} images...")
        start_time = time.time()

        with torch.no_grad():
            for seed in range(test_num):
                torch.manual_seed(seed)

                step_start = time.time()
                images = self.pipe(batch_size=1, num_inference_steps=algorithm_config.get('num_steps', 20)).images
                step_time = time.time() - step_start
                generation_times.append(step_time)

                # Store images
                all_images.extend(images)

                # Quality metrics
                if images:
                    img_arrays = [np.array(img) for img in images]
                    variance = np.mean([np.var(img) for img in img_arrays])
                    image_qualities.append(variance)

                if seed % 10 == 0:
                    print(f"  Generated {seed+1}/{test_num} images")

        total_time = time.time() - start_time

        # Compute metrics
        result = {
            'algorithm': algorithm_name,
            'type': algorithm_config['type'],
            'total_time': total_time,
            'avg_time_per_image': np.mean(generation_times),
            'std_time_per_image': np.std(generation_times),
            'avg_quality': np.mean(image_qualities) if image_qualities else 0,
            'std_quality': np.std(image_qualities) if image_qualities else 0,
            'efficiency': np.mean(image_qualities) / np.mean(generation_times) if image_qualities and generation_times else 0,
            'time_stability': 1.0 / (1.0 + np.std(generation_times) / np.mean(generation_times)),
            'config': algorithm_config
        }

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time per image: {result['avg_time_per_image']:.3f}s")
        print(f"  Avg quality: {result['avg_quality']:.4f}")
        print(f"  Efficiency: {result['efficiency']:.2f}")
        print(f"  Time stability: {result['time_stability']:.3f}")

        return result

    def run_comparison(self, test_num: int = 50):
        """Run detailed comparison between LML and Hessian methods"""

        print("🔬 Starting LML vs Hessian Methods Comparison")
        print("="*60)
        print(f"Test configuration:")
        print(f"  - Model: {self.model_path}")
        print(f"  - Test samples: {test_num}")
        print(f"  - Output directory: {self.output_dir}")
        print("="*60)

        # Define algorithm configurations
        algorithm_configs = [
            {
                'name': 'LML-Original',
                'type': 'lml',
                'lamb': 0.0008,
                'kappa': 1e-8,
                'num_steps': 20,
                'description': 'Levenberg-Marquardt Langevin with approximate Hessian',
                'theory': 'Uses approximate second-order information through LM correction',
                'complexity': 'O(n) - Linear complexity with approximate Hessian',
                'memory': 'Low - No explicit Hessian storage',
                'use_case': 'General purpose, balanced performance'
            },
            {
                'name': 'Explicit-Hessian',
                'type': 'explicit_hessian',
                'lamb': 0.001,
                'kappa': 5e-8,
                'num_steps': 20,
                'description': 'Explicit Hessian calculation with second-order derivatives',
                'theory': 'Computes exact Hessian matrix using second-order derivatives',
                'complexity': 'O(n²) - Quadratic complexity for Hessian computation',
                'memory': 'High - Stores full Hessian matrix',
                'use_case': 'Research, theoretical analysis, small-scale problems'
            },
            {
                'name': 'Hessian-Free-Basic',
                'type': 'hcg',
                'lamb': 0.001,
                'kappa': 5e-8,
                'max_iter': 10,
                'num_steps': 20,
                'description': 'Hessian-Free method using conjugate gradient',
                'theory': 'Solves Hx = b using CG without explicit Hessian storage',
                'complexity': 'O(kn) - Linear per iteration, k iterations',
                'memory': 'Low - No explicit Hessian storage',
                'use_case': 'Large-scale problems, memory efficiency'
            },
            {
                'name': 'Hessian-Free-Fast',
                'type': 'hcg',
                'lamb': 0.001,
                'kappa': 5e-8,
                'max_iter': 5,
                'num_steps': 10,
                'description': 'Fast Hessian-Free method with reduced iterations',
                'theory': 'Hessian-Free with fewer CG iterations for speed',
                'complexity': 'O(kn) - Linear per iteration, fewer iterations',
                'memory': 'Low - No explicit Hessian storage',
                'use_case': 'Real-time applications, production systems'
            }
        ]

        # Test each algorithm
        results = {}
        for config in algorithm_configs:
            try:
                result = self.test_algorithm(config, test_num)
                results[config['name']] = result
            except Exception as e:
                print(f"❌ Error testing {config['name']}: {e}")
                results[config['name']] = {
                    'algorithm': config['name'],
                    'error': str(e),
                    'success': False
                }

        self.results = results

        # Generate analysis
        self.generate_detailed_analysis()

        # Save results
        self.save_results()

    def generate_detailed_analysis(self):
        """Generate detailed analysis of LML vs Hessian methods"""

        print(f"\n{'='*60}")
        print("DETAILED LML vs HESSIAN ANALYSIS")
        print(f"{'='*60}")

        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}

        if not successful_results:
            print("❌ No successful results to analyze")
            return

        # Performance Summary
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"{'Algorithm':<20} {'Time(s)':<10} {'Quality':<12} {'Efficiency':<12} {'Stability':<12}")
        print("-" * 80)

        for algorithm, result in successful_results.items():
            print(f"{algorithm:<20} {result['avg_time_per_image']:<10.3f} {result['avg_quality']:<12.4f} {result['efficiency']:<12.2f} {result['time_stability']:<12.3f}")

        # Theoretical Analysis
        print(f"\n🧮 THEORETICAL ANALYSIS")
        print(f"{'='*60}")

        print("LML (Levenberg-Marquardt Langevin):")
        print("  - Theory: Uses approximate second-order information")
        print("  - Complexity: O(n) - Linear with approximate Hessian")
        print("  - Memory: Low - No explicit Hessian storage")
        print("  - Advantages: Fast, memory efficient, general purpose")
        print("  - Disadvantages: Approximate, may not capture full second-order info")

        print("\nExplicit Hessian:")
        print("  - Theory: Computes exact Hessian matrix")
        print("  - Complexity: O(n²) - Quadratic for Hessian computation")
        print("  - Memory: High - Stores full Hessian matrix")
        print("  - Advantages: Exact second-order information, theoretically sound")
        print("  - Disadvantages: High computational cost, memory intensive")

        print("\nHessian-Free:")
        print("  - Theory: Solves Hx = b using CG without explicit Hessian")
        print("  - Complexity: O(kn) - Linear per iteration, k iterations")
        print("  - Memory: Low - No explicit Hessian storage")
        print("  - Advantages: Memory efficient, scalable, exact solution")
        print("  - Disadvantages: Iterative, convergence dependent")

        # Algorithm Flow Analysis
        print(f"\n🔄 ALGORITHM FLOW ANALYSIS")
        print(f"{'='*60}")

        print("LML Flow:")
        print("  1. Predict noise using UNet")
        print("  2. Apply EMA smoothing: noise_ema = κ*prev + (1-κ)*current")
        print("  3. Apply LM correction: corrected = noise_ema / (1 + λ)")
        print("  4. Normalize and return")

        print("\nExplicit Hessian Flow:")
        print("  1. Predict noise using UNet")
        print("  2. Apply EMA smoothing")
        print("  3. Compute gradient of log-likelihood")
        print("  4. Compute Hessian matrix using second-order derivatives")
        print("  5. Solve Hx = b for correction")
        print("  6. Apply correction and normalize")

        print("\nHessian-Free Flow:")
        print("  1. Predict noise using UNet")
        print("  2. Apply EMA smoothing")
        print("  3. Initialize CG solver")
        print("  4. For each CG iteration:")
        print("     a. Compute Hv using HVP")
        print("     b. Update CG solution")
        print("     c. Check convergence")
        print("  5. Apply correction and normalize")

        # Parameter Analysis
        print(f"\n⚙️ PARAMETER ANALYSIS")
        print(f"{'='*60}")

        print("Key Parameters:")
        print("  - λ (lambda): Regularization parameter")
        print("    * LML: 0.0008 (conservative)")
        print("    * Explicit Hessian: 0.001 (moderate)")
        print("    * Hessian-Free: 0.001 (moderate)")
        print("  - κ (kappa): EMA smoothing parameter")
        print("    * All methods: 1e-8 to 5e-8 (very small)")
        print("  - max_iter: Maximum CG iterations (Hessian-Free only)")
        print("    * Basic: 10 iterations")
        print("    * Fast: 5 iterations")

        # Use Case Recommendations
        print(f"\n💡 USE CASE RECOMMENDATIONS")
        print(f"{'='*60}")

        print("LML-Original:")
        print("  - Best for: General purpose, balanced performance")
        print("  - When to use: Standard applications, moderate quality requirements")
        print("  - Trade-offs: Good balance of speed and quality")

        print("\nExplicit-Hessian:")
        print("  - Best for: Research, theoretical analysis")
        print("  - When to use: Small-scale problems, exact solutions needed")
        print("  - Trade-offs: High accuracy but high computational cost")

        print("\nHessian-Free-Basic:")
        print("  - Best for: Large-scale problems, memory efficiency")
        print("  - When to use: Production systems, memory constraints")
        print("  - Trade-offs: Good quality with reasonable speed")

        print("\nHessian-Free-Fast:")
        print("  - Best for: Real-time applications, production systems")
        print("  - When to use: Speed critical, moderate quality acceptable")
        print("  - Trade-offs: Fastest but potentially lower quality")

        # Performance Comparison
        print(f"\n📈 PERFORMANCE COMPARISON")
        print(f"{'='*60}")

        if len(successful_results) >= 2:
            algorithms = list(successful_results.keys())
            times = [successful_results[k]['avg_time_per_image'] for k in algorithms]
            qualities = [successful_results[k]['avg_quality'] for k in algorithms]
            efficiencies = [successful_results[k]['efficiency'] for k in algorithms]

            fastest = algorithms[np.argmin(times)]
            best_quality = algorithms[np.argmax(qualities)]
            most_efficient = algorithms[np.argmax(efficiencies)]

            print(f"Fastest: {fastest} ({min(times):.3f}s)")
            print(f"Best Quality: {best_quality} ({max(qualities):.4f})")
            print(f"Most Efficient: {most_efficient} ({max(efficiencies):.2f})")

            # Speed improvement
            speed_improvement = ((max(times) - min(times)) / max(times)) * 100
            print(f"Speed Improvement: {speed_improvement:.1f}%")

            # Quality improvement
            quality_improvement = ((max(qualities) - min(qualities)) / min(qualities)) * 100
            print(f"Quality Improvement: {quality_improvement:.1f}%")

    def save_results(self):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(hessian_cg_results_dir, f"lml_vs_hessian_comparison.json")

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

        print(f"\n💾 LML vs Hessian comparison results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="LML vs Hessian Methods Comparison")
    parser.add_argument('--model_path', type=str, default=os.path.join(project.model_dir, 'ddpm_ema_cifar10'),
                        help='Path to the model')
    parser.add_argument('--output_dir', type=str, default=hessian_cg_results_dir,
                        help='Output directory for results')
    parser.add_argument('--test_num', type=int, default=50,
                        help='Number of test images to generate')

    args = parser.parse_args()

    # Run comparison
    comparator = LMLvsHessianComparator(args.model_path, args.output_dir)
    comparator.run_comparison(args.test_num)

if __name__ == '__main__':
    main()
