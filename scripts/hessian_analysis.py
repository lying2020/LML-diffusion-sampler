#!/usr/bin/env python3
"""
Hessian Matrix Analysis for LML Diffusion Sampler

This script analyzes Hessian matrix properties including:
- Condition number
- Rank
- Eigenvalue distribution
- Convergence properties
"""

import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import argparse
from scipy.linalg import svd, eigh
from scipy.sparse.linalg import eigsh

sys.path.append(os.getcwd())
from diffusers import DDPMPipeline

class HessianAnalyzer:
    """Comprehensive Hessian matrix analyzer"""

    def __init__(self, device='cuda'):
        self.device = device
        self.condition_numbers = []
        self.ranks = []
        self.eigenvalues = []
        self.hessian_norms = []

    def compute_hessian_properties(self, hessian: torch.Tensor) -> Dict:
        """Compute comprehensive Hessian properties"""
        H_np = hessian.detach().cpu().numpy()

        # SVD decomposition
        U, s, Vt = svd(H_np)

        # Remove near-zero singular values
        tol = 1e-10
        s_clean = s[s > tol]

        # Condition number
        cond_num = s_clean[0] / s_clean[-1] if len(s_clean) > 1 else float('inf')

        # Rank
        rank = len(s_clean)

        # Frobenius norm
        frobenius_norm = np.linalg.norm(H_np, 'fro')

        # Spectral norm
        spectral_norm = s_clean[0] if len(s_clean) > 0 else 0

        # Eigenvalue distribution (for symmetric matrices)
        try:
            if H_np.shape[0] == H_np.shape[1]:  # Square matrix
                eigenvals = eigh(H_np, eigvals_only=True)
                eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order
            else:
                eigenvals = s_clean  # Use singular values as approximation
        except:
            eigenvals = s_clean

        return {
            'condition_number': cond_num,
            'rank': rank,
            'frobenius_norm': frobenius_norm,
            'spectral_norm': spectral_norm,
            'eigenvalues': eigenvals,
            'singular_values': s_clean
        }

    def analyze_convergence(self, residuals: List[float]) -> Dict:
        """Analyze convergence properties"""
        if not residuals:
            return {}

        residuals = np.array(residuals)

        # Convergence rate
        if len(residuals) > 1:
            convergence_rate = np.mean(np.diff(np.log(residuals)))
        else:
            convergence_rate = 0

        # Final residual
        final_residual = residuals[-1]

        # Convergence factor
        if len(residuals) > 2:
            convergence_factor = residuals[-1] / residuals[-2]
        else:
            convergence_factor = 1

        return {
            'convergence_rate': convergence_rate,
            'final_residual': final_residual,
            'convergence_factor': convergence_factor,
            'iterations_to_converge': len(residuals)
        }

def compute_hessian_finite_diff(model, x: torch.Tensor, t: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Compute Hessian using finite differences"""
    batch_size, channels, height, width = x.shape
    total_params = batch_size * channels * height * width

    print(f"Computing Hessian for {total_params} parameters using finite differences...")

    # Flatten for computation
    x_flat = x.view(batch_size, -1)
    hessian = torch.zeros(total_params, total_params, device=x.device)

    # Compute gradient at current point
    def compute_grad(x_input):
        x_grad = x_input.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            noise_pred = model(x_grad, t)
            if hasattr(noise_pred, 'sample'):
                noise_pred = noise_pred.sample
            log_prob = -0.5 * torch.sum(noise_pred ** 2, dim=(1, 2, 3))
            log_prob = log_prob.sum()
        return torch.autograd.grad(log_prob, x_grad, create_graph=True)[0]

    grad_current = compute_grad(x)
    grad_current_flat = grad_current.view(batch_size, -1)

    # Compute Hessian row by row
    for i in range(min(total_params, 1000)):  # Limit for memory
        if i % 100 == 0:
            print(f"Computing row {i}/{min(total_params, 1000)}")

        # Create perturbation
        eps_vec = torch.zeros_like(x_flat)
        eps_vec[0, i] = eps

        # Perturb input
        x_perturbed = x_flat + eps_vec
        x_perturbed = x_perturbed.view(batch_size, channels, height, width)

        # Compute gradient at perturbed point
        grad_perturbed = compute_grad(x_perturbed)
        grad_perturbed_flat = grad_perturbed.view(batch_size, -1)

        # Finite difference
        hessian_row = (grad_perturbed_flat - grad_current_flat) / eps
        hessian[i, :] = hessian_row[0]

    return hessian

def hessian_vector_product(model, x: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute Hessian-vector product using Pearlmutter's method"""
    x_grad = x.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        noise_pred = model(x_grad, t)
        if hasattr(noise_pred, 'sample'):
            noise_pred = noise_pred.sample
        log_prob = -0.5 * torch.sum(noise_pred ** 2, dim=(1, 2, 3))
        log_prob = log_prob.sum()

    # First gradient
    grad = torch.autograd.grad(log_prob, x_grad, create_graph=True)[0]

    # Hv = ∇(∇f · v)
    grad_dot_v = torch.sum(grad * v)
    hv = torch.autograd.grad(grad_dot_v, x_grad, retain_graph=True)[0]

    return hv

def conjugate_gradient_with_analysis(model, x: torch.Tensor, t: torch.Tensor,
                                   b: torch.Tensor, max_iter: int = 50,
                                   tol: float = 1e-6) -> Tuple[torch.Tensor, List[float]]:
    """Solve Hx = b using CG with convergence analysis"""
    batch_size, channels, height, width = x.shape
    b_flat = b.view(batch_size, -1)

    # Initialize
    x_cg = torch.zeros_like(b_flat)
    r = b_flat.clone()
    p = r.clone()

    residuals = []
    r_norm_sq = torch.sum(r ** 2)
    r_norm_0 = torch.sqrt(r_norm_sq)
    residuals.append(r_norm_0.item())

    print(f"Starting CG with {max_iter} max iterations, tolerance {tol}")
    print(f"Initial residual norm: {r_norm_0.item():.6f}")

    for i in range(max_iter):
        # Convert p to proper shape for HVP
        p_reshaped = p.view(batch_size, channels, height, width)

        # Compute Hp
        Hp = hessian_vector_product(model, x, t, p_reshaped)
        Hp_flat = Hp.view(batch_size, -1)

        # Compute step size
        p_Hp = torch.sum(p * Hp_flat)
        if p_Hp <= 0:
            print(f"CG stopped at iteration {i}: p^T H p <= 0")
            break

        alpha = r_norm_sq / p_Hp

        # Update solution and residual
        x_cg = x_cg + alpha * p
        r = r - alpha * Hp_flat

        # Check convergence
        r_norm_sq_new = torch.sum(r ** 2)
        r_norm = torch.sqrt(r_norm_sq_new)
        residuals.append(r_norm.item())

        if i % 5 == 0:
            print(f"Iteration {i}: residual norm = {r_norm.item():.6f}")

        if r_norm < tol * r_norm_0:
            print(f"CG converged at iteration {i}")
            break

        # Update search direction
        beta = r_norm_sq_new / r_norm_sq
        p = r + beta * p
        r_norm_sq = r_norm_sq_new

    return x_cg.view(batch_size, channels, height, width), residuals

def run_hessian_analysis(model_id: str, test_samples: int = 3, device: str = 'cuda'):
    """Run comprehensive Hessian analysis"""

    print("🔬 Starting Comprehensive Hessian Analysis")
    print("="*60)

    # Load model
    print("Loading model...")
    pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
    pipe.unet.to(device)

    # Initialize analyzer
    analyzer = HessianAnalyzer(device)

    # Generate test samples
    print(f"\nGenerating {test_samples} test samples...")
    with torch.no_grad():
        x = torch.randn(test_samples, 3, 32, 32, device=device)
        t = torch.randint(0, 1000, (test_samples,), device=device)

    print(f"Test samples shape: {x.shape}")
    print(f"Timesteps: {t}")

    # Analysis 1: Explicit Hessian Computation
    print(f"\n{'='*60}")
    print("ANALYSIS 1: Explicit Hessian Properties")
    print(f"{'='*60}")

    for i in range(min(2, test_samples)):  # Limit for memory
        print(f"\nAnalyzing sample {i+1}/{min(2, test_samples)}")

        x_sample = x[i:i+1]
        t_sample = t[i:i+1]

        try:
            # Compute Hessian
            start_time = time.time()
            hessian = compute_hessian_finite_diff(pipe.unet, x_sample, t_sample)
            compute_time = time.time() - start_time

            # Analyze properties
            properties = analyzer.compute_hessian_properties(hessian)

            print(f"  Hessian computation time: {compute_time:.2f}s")
            print(f"  Condition number: {properties['condition_number']:.2e}")
            print(f"  Rank: {properties['rank']}")
            print(f"  Frobenius norm: {properties['frobenius_norm']:.2e}")
            print(f"  Spectral norm: {properties['spectral_norm']:.2e}")

            # Store results
            analyzer.condition_numbers.append(properties['condition_number'])
            analyzer.ranks.append(properties['rank'])
            analyzer.hessian_norms.append(properties['frobenius_norm'])
            analyzer.eigenvalues.append(properties['eigenvalues'])

        except Exception as e:
            print(f"  Error in Hessian computation: {e}")

    # Analysis 2: Hessian-Free Method Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS 2: Hessian-Free Method (CG + HVP)")
    print(f"{'='*60}")

    for i in range(min(2, test_samples)):
        print(f"\nAnalyzing sample {i+1}/{min(2, test_samples)}")

        x_sample = x[i:i+1]
        t_sample = t[i:i+1]

        try:
            # Create random target
            b = torch.randn_like(x_sample)

            # Solve using CG with analysis
            start_time = time.time()
            x_solution, residuals = conjugate_gradient_with_analysis(
                pipe.unet, x_sample, t_sample, b, max_iter=30, tol=1e-4
            )
            solve_time = time.time() - start_time

            # Analyze convergence
            convergence_analysis = analyzer.analyze_convergence(residuals)

            print(f"  CG solve time: {solve_time:.2f}s")
            print(f"  Iterations: {convergence_analysis['iterations_to_converge']}")
            print(f"  Final residual: {convergence_analysis['final_residual']:.6f}")
            print(f"  Convergence rate: {convergence_analysis['convergence_rate']:.4f}")

            # Verify solution
            Hx_approx = hessian_vector_product(pipe.unet, x_sample, t_sample, x_solution)
            residual_norm = torch.norm(Hx_approx - b)
            relative_error = residual_norm / torch.norm(b)

            print(f"  Verification residual: {residual_norm.item():.6f}")
            print(f"  Relative error: {relative_error.item():.6f}")

        except Exception as e:
            print(f"  Error in Hessian-Free analysis: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")

    if analyzer.condition_numbers:
        print(f"Average condition number: {np.mean(analyzer.condition_numbers):.2e}")
        print(f"Average rank: {np.mean(analyzer.ranks):.0f}")
        print(f"Average Hessian norm: {np.mean(analyzer.hessian_norms):.2e}")

        # Condition number analysis
        cond_nums = np.array(analyzer.condition_numbers)
        print(f"Condition number range: {np.min(cond_nums):.2e} - {np.max(cond_nums):.2e}")

        if np.any(cond_nums > 1e12):
            print("⚠️  High condition numbers detected - matrix may be ill-conditioned")
        elif np.any(cond_nums > 1e6):
            print("⚠️  Moderate condition numbers - some numerical instability possible")
        else:
            print("✅ Good condition numbers - matrix is well-conditioned")

    print(f"\n🎯 Key Insights:")
    print(f"  - Hessian matrices in diffusion models can be ill-conditioned")
    print(f"  - Hessian-Free methods provide efficient alternatives to explicit computation")
    print(f"  - CG convergence depends on condition number and problem structure")
    print(f"  - Both methods can be integrated into LML for improved performance")

def main():
    parser = argparse.ArgumentParser(description="Hessian matrix analysis")
    parser.add_argument('--model_id', type=str, default='./model/ddpm_ema_cifar10',
                        help='Path to the model')
    parser.add_argument('--test_samples', type=int, default=3,
                        help='Number of test samples')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Get absolute path
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_id = os.path.join(project_dir, args.model_id)

    run_hessian_analysis(model_id, args.test_samples, args.device)

if __name__ == '__main__':
    main()
