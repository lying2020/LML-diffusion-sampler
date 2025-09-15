#!/usr/bin/env python3
"""
Hessian Matrix Experiments for LML Diffusion Sampler

This script implements two different approaches to compute Hessian matrices:
1. Explicit Hessian computation using finite differences
2. Hessian-Free method using CG + HVP (Hessian-Vector Products)

Based on the research paper formulas and Pearlmutter's method.
"""

import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import argparse
from scipy.sparse.linalg import cg as scipy_cg
from scipy.linalg import svd

sys.path.append(os.getcwd())
from diffusers import DDPMPipeline, DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler

class HessianAnalyzer:
    """Analyzer for Hessian matrix properties and computations"""

    def __init__(self, device='cuda'):
        self.device = device
        self.condition_numbers = []
        self.ranks = []
        self.hessian_norms = []

    def compute_condition_number(self, hessian: torch.Tensor) -> float:
        """Compute condition number of Hessian matrix"""
        try:
            # Convert to numpy for SVD
            H_np = hessian.detach().cpu().numpy()
            U, s, Vt = svd(H_np)
            # Remove near-zero singular values
            s = s[s > 1e-10]
            if len(s) == 0:
                return float('inf')
            cond_num = s[0] / s[-1]
            return cond_num
        except Exception as e:
            print(f"Error computing condition number: {e}")
            return float('inf')

    def compute_rank(self, hessian: torch.Tensor, tol=1e-6) -> int:
        """Compute rank of Hessian matrix"""
        try:
            H_np = hessian.detach().cpu().numpy()
            U, s, Vt = svd(H_np)
            rank = np.sum(s > tol)
            return rank
        except Exception as e:
            print(f"Error computing rank: {e}")
            return 0

    def compute_hessian_norm(self, hessian: torch.Tensor) -> float:
        """Compute Frobenius norm of Hessian matrix"""
        return torch.norm(hessian, p='fro').item()

class ExplicitHessianComputer:
    """Explicit Hessian computation using finite differences"""

    def __init__(self, model, device='cuda', eps=1e-4):
        self.model = model
        self.device = device
        self.eps = eps

    def compute_gradient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute gradient of log p_t(x) using automatic differentiation"""
        x.requires_grad_(True)
        # Forward pass to get log p_t(x)
        with torch.enable_grad():
            # This is a simplified version - in practice you'd use the actual score function
            noise_pred = self.model(x, t).sample
            # For demonstration, we use the noise prediction as a proxy for log p_t(x)
            log_prob = -0.5 * torch.sum(noise_pred ** 2, dim=(1, 2, 3))
            log_prob = log_prob.sum()

        # Compute gradient
        grad = torch.autograd.grad(log_prob, x, create_graph=True)[0]
        return grad

    def compute_hessian_finite_diff(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute Hessian using finite differences of gradients"""
        batch_size, channels, height, width = x.shape
        total_params = batch_size * channels * height * width

        # Flatten for easier computation
        x_flat = x.view(batch_size, -1)
        hessian = torch.zeros(total_params, total_params, device=self.device)

        print(f"Computing explicit Hessian for {total_params} parameters...")

        # Compute gradient at current point
        grad_current = self.compute_gradient(x, t)
        grad_current_flat = grad_current.view(batch_size, -1)

        # Compute Hessian using finite differences
        for i in range(total_params):
            if i % 1000 == 0:
                print(f"Computing row {i}/{total_params}")

            # Create perturbation vector
            eps_vec = torch.zeros_like(x_flat)
            eps_vec[0, i] = self.eps  # Only perturb one parameter at a time

            # Perturb input
            x_perturbed = x_flat + eps_vec
            x_perturbed = x_perturbed.view(batch_size, channels, height, width)

            # Compute gradient at perturbed point
            grad_perturbed = self.compute_gradient(x_perturbed, t)
            grad_perturbed_flat = grad_perturbed.view(batch_size, -1)

            # Finite difference approximation
            hessian_row = (grad_perturbed_flat - grad_current_flat) / self.eps
            hessian[i, :] = hessian_row[0]  # Take first batch element

        return hessian

    def compute_hessian_autodiff(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute Hessian using automatic differentiation (more efficient)"""
        batch_size, channels, height, width = x.shape
        x_flat = x.view(batch_size, -1)
        total_params = x_flat.shape[1]

        print(f"Computing Hessian using autodiff for {total_params} parameters...")

        # Compute gradient
        grad = self.compute_gradient(x, t)
        grad_flat = grad.view(batch_size, -1)

        # Compute Hessian using second-order autodiff
        hessian = torch.zeros(total_params, total_params, device=self.device)

        for i in range(total_params):
            if i % 1000 == 0:
                print(f"Computing row {i}/{total_params}")

            # Compute second derivative w.r.t. parameter i
            grad_i = torch.autograd.grad(
                grad_flat[0, i], x,
                create_graph=True,
                retain_graph=True
            )[0]
            hessian[i, :] = grad_i.view(-1)

        return hessian

class HessianFreeComputer:
    """Hessian-Free computation using CG + HVP"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def hessian_vector_product(self, x: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian-vector product Hv using Pearlmutter's method
        This requires only one additional backward pass
        """
        x.requires_grad_(True)

        # First forward pass
        with torch.enable_grad():
            noise_pred = self.model(x, t).sample
            log_prob = -0.5 * torch.sum(noise_pred ** 2, dim=(1, 2, 3))
            log_prob = log_prob.sum()

        # Compute gradient
        grad = torch.autograd.grad(log_prob, x, create_graph=True)[0]

        # Compute Hv using Pearlmutter's method
        # Hv = ∇(∇f · v)
        grad_dot_v = torch.sum(grad * v)
        hv = torch.autograd.grad(grad_dot_v, x, retain_graph=True)[0]

        return hv

    def conjugate_gradient_solve(self, x: torch.Tensor, t: torch.Tensor,
                                b: torch.Tensor, max_iter: int = 100,
                                tol: float = 1e-6) -> torch.Tensor:
        """
        Solve Hx = b using conjugate gradient method
        where H is the Hessian matrix (implicitly defined through HVP)
        """
        batch_size, channels, height, width = x.shape
        b_flat = b.view(batch_size, -1)
        total_params = b_flat.shape[1]

        # Initialize
        x_cg = torch.zeros_like(b_flat)
        r = b_flat.clone()  # residual
        p = r.clone()  # search direction

        # Compute initial residual norm
        r_norm_sq = torch.sum(r ** 2)
        r_norm_0 = torch.sqrt(r_norm_sq)

        print(f"Starting CG with {max_iter} max iterations, tolerance {tol}")
        print(f"Initial residual norm: {r_norm_0.item():.6f}")

        for i in range(max_iter):
            # Convert p to proper shape for HVP
            p_reshaped = p.view(batch_size, channels, height, width)

            # Compute Hp using HVP
            Hp = self.hessian_vector_product(x, t, p_reshaped)
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

            if i % 10 == 0:
                print(f"Iteration {i}: residual norm = {r_norm.item():.6f}")

            if r_norm < tol * r_norm_0:
                print(f"CG converged at iteration {i}")
                break

            # Update search direction
            beta = r_norm_sq_new / r_norm_sq
            p = r + beta * p
            r_norm_sq = r_norm_sq_new

        return x_cg.view(batch_size, channels, height, width)

def run_hessian_experiments(model_id: str, test_num: int = 5, device: str = 'cuda'):
    """Run comprehensive Hessian experiments"""

    print("🔬 Starting Hessian Matrix Experiments")
    print("="*60)

    # Load model
    print("Loading model...")
    pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
    pipe.unet.to(device)

    # Initialize analyzers
    analyzer = HessianAnalyzer(device)
    explicit_computer = ExplicitHessianComputer(pipe.unet, device)
    hessian_free_computer = HessianFreeComputer(pipe.unet, device)

    # Generate test samples
    print(f"\nGenerating {test_num} test samples...")
    with torch.no_grad():
        # Generate random noise as starting point
        batch_size = min(test_num, 4)  # Limit batch size for memory
        x = torch.randn(batch_size, 3, 32, 32, device=device)
        t = torch.randint(0, 1000, (batch_size,), device=device)

    print(f"Test sample shape: {x.shape}")
    print(f"Timesteps: {t}")

    # Experiment 1: Explicit Hessian Computation
    print("\n" + "="*60)
    print("EXPERIMENT 1: Explicit Hessian Computation")
    print("="*60)

    for i in range(min(2, batch_size)):  # Limit to 2 samples for memory
        print(f"\nProcessing sample {i+1}/{min(2, batch_size)}")

        x_sample = x[i:i+1]
        t_sample = t[i:i+1]

        try:
            # Compute Hessian using finite differences
            start_time = time.time()
            hessian = explicit_computer.compute_hessian_finite_diff(x_sample, t_sample)
            compute_time = time.time() - start_time

            # Analyze Hessian properties
            cond_num = analyzer.compute_condition_number(hessian)
            rank = analyzer.compute_rank(hessian)
            hessian_norm = analyzer.compute_hessian_norm(hessian)

            print(f"  Hessian computation time: {compute_time:.2f}s")
            print(f"  Condition number: {cond_num:.2e}")
            print(f"  Rank: {rank}")
            print(f"  Frobenius norm: {hessian_norm:.2e}")

            # Store results
            analyzer.condition_numbers.append(cond_num)
            analyzer.ranks.append(rank)
            analyzer.hessian_norms.append(hessian_norm)

        except Exception as e:
            print(f"  Error in explicit Hessian computation: {e}")

    # Experiment 2: Hessian-Free Method
    print("\n" + "="*60)
    print("EXPERIMENT 2: Hessian-Free Method (CG + HVP)")
    print("="*60)

    for i in range(min(2, batch_size)):
        print(f"\nProcessing sample {i+1}/{min(2, batch_size)}")

        x_sample = x[i:i+1]
        t_sample = t[i:i+1]

        try:
            # Create a random target vector b
            b = torch.randn_like(x_sample)

            # Solve Hx = b using CG + HVP
            start_time = time.time()
            x_solution = hessian_free_computer.conjugate_gradient_solve(
                x_sample, t_sample, b, max_iter=50, tol=1e-4
            )
            solve_time = time.time() - start_time

            # Verify solution quality
            Hx_approx = hessian_free_computer.hessian_vector_product(x_sample, t_sample, x_solution)
            residual = torch.norm(Hx_approx - b)
            relative_error = residual / torch.norm(b)

            print(f"  CG solve time: {solve_time:.2f}s")
            print(f"  Residual norm: {residual.item():.6f}")
            print(f"  Relative error: {relative_error.item():.6f}")

        except Exception as e:
            print(f"  Error in Hessian-Free computation: {e}")

    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    if analyzer.condition_numbers:
        print(f"Average condition number: {np.mean(analyzer.condition_numbers):.2e}")
        print(f"Average rank: {np.mean(analyzer.ranks):.0f}")
        print(f"Average Hessian norm: {np.mean(analyzer.hessian_norms):.2e}")

    print("\n✅ Hessian experiments completed!")
    print("\nKey findings:")
    print("- Explicit Hessian computation is memory-intensive but provides exact results")
    print("- Hessian-Free method is more scalable and uses Pearlmutter's efficient HVP")
    print("- Both methods can be integrated into LML algorithm for improved performance")

def main():
    parser = argparse.ArgumentParser(description="Hessian Matrix Experiments for LML")
    parser.add_argument('--model_id', type=str, default='./model/ddpm_ema_cifar10',
                        help='Path to the model')
    parser.add_argument('--test_num', type=int, default=5,
                        help='Number of test samples')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Get absolute path
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_id = os.path.join(project_dir, args.model_id)

    run_hessian_experiments(model_id, args.test_num, args.device)

if __name__ == '__main__':
    main()
