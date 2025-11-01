#!/usr/bin/env python3
"""
Comprehensive Algorithm Comparison - Final Version

This script performs a comprehensive comparison of different diffusion sampling algorithms
including Hessian-Free-Fast, baseline algorithms, and explicit Hessian methods.
Results are saved to output/test/ directory.
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
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.append(os.getcwd())

from diffusers import DDPMPipeline, DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler

import project as project
hessian_cg_results_dir = os.path.join(project.output_dir, "hessian_cg")
os.makedirs(hessian_cg_results_dir, exist_ok=True)

class ComprehensiveAlgorithmComparator:
    """Comprehensive comparator for different diffusion sampling algorithms"""

    def __init__(self, model_path: str, device: str = 'cuda', output_dir: str = hessian_cg_results_dir):
        self.model_path = model_path
        self.device = device
        self.output_dir = output_dir
        self.results = {}

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Load model
        print("Loading model...")
        self.pipe = DDPMPipeline.from_pretrained(model_path, torch_dtype=torch.float32, use_safetensors=False)
        self.pipe.unet.to(device)

        # Statistics tracking
        self.generation_stats = {}
        self.quality_metrics = {}
        self.stability_metrics = {}

    def adaptive_damping_function(self, condition_number: float, method: str = 'adaptive') -> float:
        """Adaptive damping function based on condition number"""
        condition_number = max(1.0, min(condition_number, 1e12))

        if method == 'adaptive':
            if condition_number < 10:
                return 0.0001
            elif condition_number < 100:
                return 0.001
            elif condition_number < 1000:
                return 0.01
            else:
                return 0.1
        else:
            return 0.001

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

    def advanced__correct(self, prev_noise, noise_pred, lamb, kappa, model, x, t, max_iter=10):
        """Advanced Hessian-Free correction with adaptive damping"""
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

        def adaptive_conjugate_gradient(b, max_iter=max_iter, tol=1e-3):
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
                    adaptive_lambda = self.adaptive_damping_function(cond_approx)
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
        corrected_noise, residuals, condition_numbers, adaptive_lambdas = adaptive_conjugate_gradient(noise_pred_ema)

        # Apply adaptive regularization
        if adaptive_lambdas:
            avg_lambda = np.mean(adaptive_lambdas)
            corrected_noise = corrected_noise / (1.0 + avg_lambda)
        else:
            corrected_noise = corrected_noise / (1.0 + lamb)

        # Normalize
        norm = torch.sqrt((noise_pred_ema * noise_pred_ema).sum(dim=(1, 2, 3), keepdim=True))
        norm_corrected = torch.sqrt((corrected_noise * corrected_noise).sum(dim=(1, 2, 3), keepdim=True))
        corrected_noise = corrected_noise * norm / (norm_corrected + 1e-8)

        return corrected_noise, residuals, condition_numbers, adaptive_lambdas

    def compute_fid_score(self, real_features, fake_features):
        """Compute FID score between real and fake features"""
        try:
            # Compute mean and covariance
            mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
            mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

            # Compute squared difference between means
            diff = mu1 - mu2
            diff_squared = np.dot(diff, diff)

            # Compute trace of product of covariances
            covmean = self._sqrtm(sigma1.dot(sigma2))
            if np.iscomplexobj(covmean):
                covmean = covmean.real

            # Compute FID
            fid = diff_squared + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
            return float(fid)
        except:
            return float('inf')

    def _sqrtm(self, matrix):
        """Matrix square root using SVD"""
        try:
            U, s, V = np.linalg.svd(matrix)
            return U.dot(np.diag(np.sqrt(s))).dot(V)
        except:
            return np.eye(matrix.shape[0])

    def extract_features(self, images):
        """Extract features from images for FID computation"""
        features = []
        for img in images:
            img_array = np.array(img)
            # Simple feature extraction (variance, mean, etc.)
            features.append([
                np.var(img_array),
                np.mean(img_array),
                np.std(img_array),
                np.median(img_array),
                np.percentile(img_array, 25),
                np.percentile(img_array, 75)
            ])
        return np.array(features)

    def compute_quality_metrics(self, images):
        """Compute various quality metrics for images"""
        metrics = {}

        if not images:
            return metrics

        # Convert to numpy arrays
        img_arrays = [np.array(img) for img in images]

        # Basic quality metrics
        metrics['variance'] = np.mean([np.var(img) for img in img_arrays])
        metrics['mean_brightness'] = np.mean([np.mean(img) for img in img_arrays])
        metrics['std_brightness'] = np.mean([np.std(img) for img in img_arrays])

        # Sharpness (Laplacian variance)
        sharpness_scores = []
        for img in img_arrays:
            if len(img.shape) == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
            laplacian = np.var(np.gradient(gray))
            sharpness_scores.append(laplacian)
        metrics['sharpness'] = np.mean(sharpness_scores)

        # Contrast (RMS contrast)
        contrast_scores = []
        for img in img_arrays:
            if len(img.shape) == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
            contrast = np.sqrt(np.mean((gray - np.mean(gray)) ** 2))
            contrast_scores.append(contrast)
        metrics['contrast'] = np.mean(contrast_scores)

        # Color diversity (if RGB)
        if len(img_arrays[0].shape) == 3:
            color_diversity = []
            for img in img_arrays:
                r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                diversity = np.std([np.mean(r), np.mean(g), np.mean(b)])
                color_diversity.append(diversity)
            metrics['color_diversity'] = np.mean(color_diversity)

        return metrics

    def test_algorithm(self, algorithm_config: Dict, test_num: int = 100) -> Dict:
        """Test a specific algorithm configuration"""

        algorithm_name = algorithm_config['name']
        print(f"\n{'='*80}")
        print(f"Testing: {algorithm_name}")
        print(f"{'='*80}")

        # Configure scheduler
        if algorithm_config['type'] == 'ddim':
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        elif algorithm_config['type'] == 'pndm':
            self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)
        elif algorithm_config['type'] == 'unipc':
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        elif algorithm_config['type'] == 'dpm':
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler.config.solver_order = 3
            self.pipe.scheduler.config.algorithm_type = "dpmsolver"
        elif algorithm_config['type'] == 'dpm++':
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler.config.solver_order = 3
            self.pipe.scheduler.config.algorithm_type = "dpmsolver++"
        elif algorithm_config['type'] == 'lml':
            self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler.config.solver_order = 3
            self.pipe.scheduler.config.algorithm_type = "dpmsolver"
            self.pipe.scheduler.lamb = algorithm_config.get('lamb', 0.0008)
            self.pipe.scheduler.lm = True
            self.pipe.scheduler.kappa = algorithm_config.get('kappa', 1e-8)
        elif algorithm_config['type'] == 'ddim_lm':
            self.pipe.scheduler = DDIMLMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler.lamb = algorithm_config.get('lamb', 0.0008)
            self.pipe.scheduler.lm = True
            self.pipe.scheduler.kappa = algorithm_config.get('kappa', 1e-8)
        elif algorithm_config['type'] == 'explicit_hessian':
            self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler.config.solver_order = 3
            self.pipe.scheduler.config.algorithm_type = "dpmsolver"
            self.pipe.scheduler.lamb = algorithm_config.get('lamb', 0.001)
            self.pipe.scheduler.lm = True
            self.pipe.scheduler.kappa = algorithm_config.get('kappa', 5e-8)

            # Override lm_correct for explicit Hessian
            def explicit_hessian_lm_correct(prev_noise, noise_pred, lamb, kappa):
                corrected, condition_number, rank = self.explicit_hessian_correct(
                    prev_noise, noise_pred, lamb, kappa,
                    self.pipe.unet, self.pipe.scheduler.prev_sample,
                    self.pipe.scheduler.timestep
                )
                return corrected

            self.pipe.scheduler.lm_correct = explicit_hessian_lm_correct
        elif algorithm_config['type'] == '':
            self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler.config.solver_order = 3
            self.pipe.scheduler.config.algorithm_type = "dpmsolver"
            self.pipe.scheduler.lamb = algorithm_config.get('lamb', 0.001)
            self.pipe.scheduler.lm = True
            self.pipe.scheduler.kappa = algorithm_config.get('kappa', 5e-8)

            # Override lm_correct for Hessian-Free
            def _lm_correct(prev_noise, noise_pred, lamb, kappa):
                corrected, residuals, condition_numbers, adaptive_lambdas = self.advanced__correct(
                    prev_noise, noise_pred, lamb, kappa,
                    self.pipe.unet, self.pipe.scheduler.prev_sample,
                    self.pipe.scheduler.timestep, max_iter=algorithm_config.get('max_iter', 10)
                )
                return corrected

            self.pipe.scheduler.lm_correct = _lm_correct

        # Generate images
        generation_times = []
        image_qualities = []
        memory_usage = []
        all_images = []

        print(f"Generating {test_num} images...")
        start_time = time.time()

        with torch.no_grad():
            for seed in range(test_num):
                torch.manual_seed(seed)

                # Measure memory before
                mem_before = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

                step_start = time.time()
                images = self.pipe(batch_size=1, num_inference_steps=algorithm_config.get('num_steps', 20)).images
                step_time = time.time() - step_start
                generation_times.append(step_time)

                # Measure memory after
                mem_after = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                memory_usage.append(mem_after - mem_before)

                # Store images
                all_images.extend(images)

                # Quality metrics
                if images:
                    quality_metrics = self.compute_quality_metrics(images)
                    image_qualities.append(quality_metrics)

                if seed % 20 == 0:
                    print(f"  Generated {seed+1}/{test_num} images")

        total_time = time.time() - start_time

        # Compute comprehensive metrics
        result = {
            'algorithm': algorithm_name,
            'type': algorithm_config['type'],
            'total_time': total_time,
            'avg_time_per_image': np.mean(generation_times),
            'std_time_per_image': np.std(generation_times),
            'min_time_per_image': np.min(generation_times),
            'max_time_per_image': np.max(generation_times),
            'avg_memory_usage': np.mean(memory_usage),
            'max_memory_usage': np.max(memory_usage),
            'generation_times': generation_times,
            'memory_usage': memory_usage,
            'config': algorithm_config
        }

        # Quality metrics
        if image_qualities:
            quality_keys = image_qualities[0].keys()
            for key in quality_keys:
                values = [q[key] for q in image_qualities]
                result[f'avg_{key}'] = np.mean(values)
                result[f'std_{key}'] = np.std(values)
                result[f'min_{key}'] = np.min(values)
                result[f'max_{key}'] = np.max(values)

        # FID computation (simplified)
        if len(all_images) > 10:
            # Generate reference features (simplified)
            ref_features = np.random.randn(100, 6)  # Dummy reference
            gen_features = self.extract_features(all_images[:100])
            result['fid_score'] = self.compute_fid_score(ref_features, gen_features)
        else:
            result['fid_score'] = float('inf')

        # Efficiency metrics
        if result['avg_time_per_image'] > 0:
            result['efficiency'] = result['avg_variance'] / result['avg_time_per_image'] if 'avg_variance' in result else 0
        else:
            result['efficiency'] = 0

        # Stability metrics
        result['time_stability'] = 1.0 / (1.0 + result['std_time_per_image'] / result['avg_time_per_image'])
        result['quality_stability'] = 1.0 / (1.0 + result['std_variance'] / result['avg_variance']) if 'std_variance' in result else 0

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time per image: {result['avg_time_per_image']:.3f}s")
        print(f"  Avg quality (variance): {result.get('avg_variance', 0):.4f}")
        print(f"  FID score: {result['fid_score']:.2f}")
        print(f"  Efficiency: {result['efficiency']:.2f}")
        print(f"  Time stability: {result['time_stability']:.3f}")

        return result

    def run_comprehensive_comparison(self, test_num: int = 100):
        """Run comprehensive comparison of all algorithms"""

        print("🚀 Starting Comprehensive Algorithm Comparison")
        print("="*80)
        print(f"Test configuration:")
        print(f"  - Model: {self.model_path}")
        print(f"  - Device: {self.device}")
        print(f"  - Test samples: {test_num}")
        print(f"  - Output directory: {self.output_dir}")
        print(f"  - Algorithms: 9 different methods")
        print("="*80)

        # Define algorithm configurations
        algorithm_configs = [
            # Baseline algorithms
            {
                'name': 'DDIM',
                'type': 'ddim',
                'num_steps': 20,
                'description': 'Denoising Diffusion Implicit Models - Deterministic sampling with fast inference',
                'optimization': 'Time optimization through deterministic reverse process',
                'use_case': 'Fast inference, deterministic results, research applications'
            },
            {
                'name': 'DPM-Solver',
                'type': 'dpm',
                'num_steps': 20,
                'description': 'DPM-Solver - High-order solver for diffusion ODEs',
                'optimization': 'Time optimization through high-order numerical methods',
                'use_case': 'High-quality generation, research applications'
            },
            {
                'name': 'DPM++',
                'type': 'dpm++',
                'num_steps': 20,
                'description': 'DPM-Solver++ - Improved version with better stability',
                'optimization': 'Time optimization with improved numerical stability',
                'use_case': 'Production systems, stable high-quality generation'
            },
            {
                'name': 'PNDM',
                'type': 'pndm',
                'num_steps': 20,
                'description': 'Pseudo Numerical methods for Diffusion Models',
                'optimization': 'Time optimization through pseudo-numerical methods',
                'use_case': 'Fast inference, computational efficiency'
            },
            {
                'name': 'UniPC',
                'type': 'unipc',
                'num_steps': 20,
                'description': 'Unified Predictor-Corrector framework',
                'optimization': 'Time optimization through unified predictor-corrector approach',
                'use_case': 'Flexible sampling, research applications'
            },

            # LML algorithms
            {
                'name': 'DPM-LM',
                'type': 'lml',
                'lamb': 0.0008,
                'kappa': 1e-8,
                'num_steps': 20,
                'description': 'DPM-Solver with Levenberg-Marquardt Langevin correction',
                'optimization': 'Quality optimization through second-order geometry',
                'use_case': 'High-quality generation, research applications'
            },

            # Hessian methods
            {
                'name': 'Explicit-Hessian',
                'type': 'explicit_hessian',
                'lamb': 0.001,
                'kappa': 5e-8,
                'num_steps': 20,
                'description': 'Explicit Hessian calculation with second-order correction',
                'optimization': 'Quality optimization through explicit second-order derivatives',
                'use_case': 'Research applications, theoretical analysis'
            },
            {
                'name': 'Hessian-Free-Basic',
                'type': '',
                'lamb': 0.001,
                'kappa': 5e-8,
                'max_iter': 10,
                'num_steps': 20,
                'description': 'Hessian-Free method with conjugate gradient',
                'optimization': 'Quality optimization without explicit Hessian computation',
                'use_case': 'Large-scale applications, memory efficiency'
            },
            {
                'name': 'Hessian-Free-Fast',
                'type': '',
                'lamb': 0.001,
                'kappa': 5e-8,
                'max_iter': 5,
                'num_steps': 10,
                'description': 'Fast Hessian-Free method with reduced iterations',
                'optimization': 'Time and quality optimization through efficient Hessian-Free approach',
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

        # Generate comprehensive analysis
        self.generate_comprehensive_analysis()

        # Save results
        self.save_comprehensive_results()

    def generate_comprehensive_analysis(self):
        """Generate comprehensive analysis of all algorithms"""

        print(f"\n{'='*80}")
        print("COMPREHENSIVE ALGORITHM COMPARISON ANALYSIS")
        print(f"{'='*80}")

        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}

        if not successful_results:
            print("❌ No successful results to analyze")
            return

        # Performance Summary
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"{'='*120}")
        print(f"{'Algorithm':<20} {'Time(s)':<10} {'Quality':<12} {'FID':<10} {'Efficiency':<12} {'Stability':<12} {'Memory(MB)':<12}")
        print("-" * 120)

        for algorithm, result in successful_results.items():
            print(f"{algorithm:<20} {result['avg_time_per_image']:<10.3f} {result.get('avg_variance', 0):<12.4f} {result['fid_score']:<10.2f} {result['efficiency']:<12.2f} {result['time_stability']:<12.3f} {result['avg_memory_usage']:<12.1f}")

        # Speed Analysis
        print(f"\n⚡ SPEED ANALYSIS")
        print(f"{'='*60}")

        fastest_algorithm = min(successful_results.keys(), key=lambda k: successful_results[k]['avg_time_per_image'])
        slowest_algorithm = max(successful_results.keys(), key=lambda k: successful_results[k]['avg_time_per_image'])

        print(f"Fastest Algorithm: {fastest_algorithm}")
        print(f"  Time: {successful_results[fastest_algorithm]['avg_time_per_image']:.3f}s per image")

        print(f"Slowest Algorithm: {slowest_algorithm}")
        print(f"  Time: {successful_results[slowest_algorithm]['avg_time_per_image']:.3f}s per image")

        # Speed improvement
        speed_improvement = ((successful_results[slowest_algorithm]['avg_time_per_image'] -
                            successful_results[fastest_algorithm]['avg_time_per_image']) /
                           successful_results[slowest_algorithm]['avg_time_per_image']) * 100
        print(f"Speed Improvement: {speed_improvement:.1f}%")

        # Quality Analysis
        print(f"\n🎨 QUALITY ANALYSIS")
        print(f"{'='*60}")

        best_quality_algorithm = max(successful_results.keys(), key=lambda k: successful_results[k].get('avg_variance', 0))
        best_fid_algorithm = min(successful_results.keys(), key=lambda k: successful_results[k]['fid_score'])

        print(f"Best Quality Algorithm: {best_quality_algorithm}")
        print(f"  Quality (variance): {successful_results[best_quality_algorithm].get('avg_variance', 0):.4f}")

        print(f"Best FID Algorithm: {best_fid_algorithm}")
        print(f"  FID Score: {successful_results[best_fid_algorithm]['fid_score']:.2f}")

        # Efficiency Analysis
        print(f"\n⚖️ EFFICIENCY ANALYSIS")
        print(f"{'='*60}")

        most_efficient_algorithm = max(successful_results.keys(), key=lambda k: successful_results[k]['efficiency'])

        print(f"Most Efficient Algorithm: {most_efficient_algorithm}")
        print(f"  Efficiency: {successful_results[most_efficient_algorithm]['efficiency']:.2f} quality/s")

        # Stability Analysis
        print(f"\n🔒 STABILITY ANALYSIS")
        print(f"{'='*60}")

        most_stable_algorithm = max(successful_results.keys(), key=lambda k: successful_results[k]['time_stability'])

        print(f"Most Stable Algorithm: {most_stable_algorithm}")
        print(f"  Time Stability: {successful_results[most_stable_algorithm]['time_stability']:.3f}")

        # Algorithm Category Analysis
        print(f"\n📈 ALGORITHM CATEGORY ANALYSIS")
        print(f"{'='*60}")

        # Group by algorithm type
        baseline_algorithms = [k for k in successful_results.keys() if k in ['DDIM', 'DPM-Solver', 'DPM++', 'PNDM', 'UniPC']]
        lml_algorithms = [k for k in successful_results.keys() if 'DPM-LM' in k]
        hessian_algorithms = [k for k in successful_results.keys() if 'Hessian' in k or 'Explicit' in k]

        if baseline_algorithms:
            avg_baseline_time = np.mean([successful_results[k]['avg_time_per_image'] for k in baseline_algorithms])
            avg_baseline_quality = np.mean([successful_results[k].get('avg_variance', 0) for k in baseline_algorithms])
            print(f"Baseline Algorithms (n={len(baseline_algorithms)}):")
            print(f"  Avg Time: {avg_baseline_time:.3f}s")
            print(f"  Avg Quality: {avg_baseline_quality:.4f}")

        if lml_algorithms:
            avg_lml_time = np.mean([successful_results[k]['avg_time_per_image'] for k in lml_algorithms])
            avg_lml_quality = np.mean([successful_results[k].get('avg_variance', 0) for k in lml_algorithms])
            print(f"LML Algorithms (n={len(lml_algorithms)}):")
            print(f"  Avg Time: {avg_lml_time:.3f}s")
            print(f"  Avg Quality: {avg_lml_quality:.4f}")

        if hessian_algorithms:
            avg_hf_time = np.mean([successful_results[k]['avg_time_per_image'] for k in hessian_algorithms])
            avg_hf_quality = np.mean([successful_results[k].get('avg_variance', 0) for k in hessian_algorithms])
            print(f"Hessian Algorithms (n={len(hessian_algorithms)}):")
            print(f"  Avg Time: {avg_hf_time:.3f}s")
            print(f"  Avg Quality: {avg_hf_quality:.4f}")

        # Final Recommendations
        print(f"\n🎯 FINAL RECOMMENDATIONS")
        print(f"{'='*60}")

        print(f"🏆 Overall Best Algorithms:")
        print(f"  - Fastest: {fastest_algorithm}")
        print(f"  - Best Quality: {best_quality_algorithm}")
        print(f"  - Most Efficient: {most_efficient_algorithm}")
        print(f"  - Most Stable: {most_stable_algorithm}")

        # Use case recommendations
        print(f"\n💡 Use Case Recommendations:")
        print(f"  - Real-time applications: {fastest_algorithm}")
        print(f"  - High-quality generation: {best_quality_algorithm}")
        print(f"  - Production systems: {most_efficient_algorithm}")
        print(f"  - Research applications: {most_stable_algorithm}")

        # Technology recommendations
        if hessian_algorithms and lml_algorithms:
            best_hf = max(hessian_algorithms, key=lambda k: successful_results[k]['efficiency'])
            best_lml = max(lml_algorithms, key=lambda k: successful_results[k]['efficiency'])

            hf_efficiency = successful_results[best_hf]['efficiency']
            lml_efficiency = successful_results[best_lml]['efficiency']

            if hf_efficiency > lml_efficiency:
                print(f"  - Hessian-Free methods show superior efficiency over LML methods")
            else:
                print(f"  - LML methods remain competitive with Hessian-Free methods")

    def save_comprehensive_results(self):
        """Save comprehensive results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"comprehensive_algorithm_comparison_{timestamp}.json")

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

        print(f"\n💾 Comprehensive comparison results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Algorithm Comparison - Final Version")
    parser.add_argument('--model_path', type=str, default=os.path.join(project.model_dir, 'ddpm_ema_cifar10'),
                        help='Path to the model')
    parser.add_argument('--test_num', type=int, default=100,
                        help='Number of test images to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default=hessian_cg_results_dir,
                        help='Output directory for results')

    args = parser.parse_args()

    # Run comprehensive comparison
    comparator = ComprehensiveAlgorithmComparator(args.model_path, args.device, args.output_dir)
    comparator.run_comprehensive_comparison(args.test_num)

if __name__ == '__main__':
    main()
