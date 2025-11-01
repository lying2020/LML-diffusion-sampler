# Copyright 2025 ZJU Lab304 Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This file is strongly influenced by https://github.com/LuChengTHU/dpm-solver

import math
import contextlib
from typing import List, Optional, Tuple, Union
import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput

# Import profiling utilities
try:
    import sys
    import os
    # Add project root to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils.profiling import profile, profile_function, get_profiler
    PROFILING_ENABLED = True
except ImportError:
    # Fallback if profiling not available
    PROFILING_ENABLED = False
    def profile(name, profiler=None):
        return contextlib.nullcontext()
    def profile_function(name=None):
        def decorator(func):
            return func
        return decorator
    def get_profiler():
        return None



def hessian_vector_product(model, x_sample: torch.Tensor, t_timestep: int, v: torch.Tensor) -> torch.Tensor:
    """
    Compute H*v using Pearlmutter's method.
    H is the Hessian of -log p_t(x), symmetrized.
    """
    # Ensure v requires grad - detach from the no_grad context
    v_grad = v.clone().detach().requires_grad_(True)

    # Save model training state
    model_training = model.training

    # Temporarily set model to eval mode for consistent behavior
    # But we still need gradients for input x_grad
    model.eval()

    try:
        # Try to disable flash attention and other optimizations that don't support second-order derivatives
        # Try new API first, then fall back to old API
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            # Use math backend which supports gradients
            sdp_context = sdpa_kernel(SDPBackend.MATH)
        except (ImportError, AttributeError):
            try:
                # Fallback to old API
                from torch.backends.cuda import sdp_kernel
                sdp_context = sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
            except (ImportError, AttributeError):
                # Final fallback: use nullcontext
                sdp_context = contextlib.nullcontext()

        # Force enable gradients - this creates a new gradient context
        # even if we're called from within a no_grad() block
        with sdp_context:
            with torch.enable_grad():
                # Create a fresh copy of x that requires grad within this gradient context
                # This ensures it's completely detached from any outer no_grad context
                x_grad = x_sample.clone().detach().requires_grad_(True)

                # Forward pass: model(x, t) - this must be inside enable_grad
                # to ensure score_pred tracks gradients w.r.t. x_grad
                # Disable any attention optimizations that might interfere
                with profile("model_forward_pass", get_profiler()):
                    score_pred = model(x_grad, t_timestep)
                if hasattr(score_pred, 'sample'):
                    score_pred = score_pred.sample

                # Verify that score_pred has gradient connection to x_grad
                if not score_pred.requires_grad:
                    # If model output doesn't require grad, we need to force it
                    # by creating a dependency
                    score_pred = score_pred + 0.0 * x_grad.sum()

                # Log probability: -0.5 * ||score||^2
                log_prob = -0.5 * torch.sum(score_pred ** 2, dim=(1, 2, 3))
                log_prob = log_prob.sum()

                # Ensure log_prob requires grad
                if not log_prob.requires_grad:
                    raise RuntimeError(
                        f"log_prob does not require grad. "
                        f"score_pred.requires_grad={score_pred.requires_grad}, "
                        f"x_grad.requires_grad={x_grad.requires_grad}"
                    )

            # First gradient: ∇log_prob w.r.t. x_grad
            # create_graph=True is needed for second-order derivatives
            with profile("first_gradient", get_profiler()):
                grad_outputs = torch.autograd.grad(
                    outputs=log_prob,
                    inputs=x_grad,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=False,
                    retain_graph=True
                )

            if len(grad_outputs) == 0 or grad_outputs[0] is None:
                raise RuntimeError("Failed to compute gradient. Check if model outputs depend on x_grad.")

            grad = grad_outputs[0]

            # Hv = ∇(∇f · v) for H = -∇²log p
            # Compute inner product: grad · v_grad
            grad_dot_v = torch.sum(grad * v_grad)

            # Check that grad_dot_v requires grad (for second derivative)
            if not grad_dot_v.requires_grad:
                raise RuntimeError(
                    f"grad_dot_v does not require grad. "
                    f"grad.requires_grad={grad.requires_grad}, "
                    f"v_grad.requires_grad={v_grad.requires_grad}, "
                    f"log_prob.requires_grad={log_prob.requires_grad}"
                )

            # Second gradient: Hv = ∇(grad · v) w.r.t. x_grad
            with profile("second_gradient", get_profiler()):
                Hv_outputs = torch.autograd.grad(
                    outputs=grad_dot_v,
                    inputs=x_grad,
                    retain_graph=False,
                    only_inputs=True,
                    allow_unused=False
                )

            if len(Hv_outputs) == 0 or Hv_outputs[0] is None:
                raise RuntimeError("Failed to compute Hessian-vector product.")

            Hv = Hv_outputs[0]

    except RuntimeError as e:
        # If we get an error about unsupported operations (e.g., flash attention),
        # fall back to finite difference approximation
        error_msg = str(e).lower()
        if "derivative" in error_msg or "not implemented" in error_msg or "scaled_dot_product" in error_msg:
            # Use finite difference approximation for Hv
            # This avoids the need for second-order derivatives
            with profile("finite_difference_hvp", get_profiler()):
                eps = 1e-4

                # Create perturbed versions of x
                x_plus = (x_sample + eps * v_grad).clone().detach().requires_grad_(True)
                x_minus = (x_sample - eps * v_grad).clone().detach().requires_grad_(True)

                with torch.enable_grad():
                    # Compute gradients at perturbed points
                    score_plus = model(x_plus, t_timestep)
                    if hasattr(score_plus, 'sample'):
                        score_plus = score_plus.sample
                    log_prob_plus = -0.5 * torch.sum(score_plus ** 2, dim=(1, 2, 3)).sum()

                    score_minus = model(x_minus, t_timestep)
                    if hasattr(score_minus, 'sample'):
                        score_minus = score_minus.sample
                    log_prob_minus = -0.5 * torch.sum(score_minus ** 2, dim=(1, 2, 3)).sum()

                # Compute gradients at both points
                grad_plus = torch.autograd.grad(log_prob_plus, x_plus, only_inputs=True, retain_graph=False)[0]
                grad_minus = torch.autograd.grad(log_prob_minus, x_minus, only_inputs=True, retain_graph=False)[0]

                # Finite difference: Hv ≈ (grad(x+εv) - grad(x-εv)) / (2ε)
                Hv = (grad_plus - grad_minus) / (2.0 * eps)
        else:
            # Re-raise if it's a different error
            raise

    finally:
        # Restore model training state
        if model_training:
            model.train()

    # Detach and return (no need to keep gradients in output)
    return Hv.detach()

# Step 4: Solve (H + λ_t I)^{-1} * noise_pred using CG
def regularized_hessian_vector_product(model, x_sample, t_timestep, v: torch.Tensor, lambda_t: float) -> torch.Tensor:
    """
    Compute (H + λ_t I) * v

    Note: This is a lightweight wrapper. Profiling is handled inside
    hessian_vector_product and at call sites.
    """
    Hv = hessian_vector_product(model, x_sample, t_timestep, v)
    # Add regularization: (H + λ_t I)v = Hv + λ_t * v
    return Hv + lambda_t * v

def conjugate_gradient_solve(model, x_sample, t_timestep, prev_solution: Optional[torch.Tensor] = None, lambda_t: float = 0.0004, b: torch.Tensor = None, cg_max_iter: int = 20, cg_tol: float = 1e-4) -> Tuple[torch.Tensor, int, float]:
    """
    Solve (H + λ_t I) * x = b using CG with optional initial guess

    Returns:
        x_cg: Solution
        num_iterations: Number of CG iterations used
        final_residual: Final residual norm (normalized by initial residual)
    """
    profiler = get_profiler()
    with profile("conjugate_gradient_solve", get_profiler()):
        # Use previous solution as initial guess if available (can improve convergence)
        if prev_solution is not None and prev_solution.shape == b.shape:
            x_cg = prev_solution.clone()
            # Compute initial residual
            # Compute initial residual (counts as 1 HVP call for warm start)
            profiler = get_profiler()
            with profile("cg_hvp_call", profiler):
                Hx = regularized_hessian_vector_product(model, x_sample, t_timestep, x_cg, lambda_t=lambda_t)
            r = b - Hx
        else:
            x_cg = torch.zeros_like(b)
            r = b.clone()
        p = r.clone()

        r_norm_sq = torch.sum(r ** 2)
        r_norm_0 = torch.sqrt(r_norm_sq)

        if r_norm_0 < 1e-10:
            return x_cg, 0, 0.0

        num_iterations = 0
        r_norm = r_norm_0  # Initialize r_norm to avoid UnboundLocalError
        for i in range(cg_max_iter):
            # CG iteration loop - each iteration makes 1 HVP call
            with profile("cg_hvp_call", profiler):
                Hp = regularized_hessian_vector_product(model, x_sample, t_timestep, p, lambda_t=lambda_t)
            p_Hp = torch.sum(p * Hp)

            if p_Hp <= 1e-10:
                # If p_Hp is too small, compute final residual before breaking
                r_norm_sq_final = torch.sum(r ** 2)
                r_norm = torch.sqrt(r_norm_sq_final)
                break

            alpha = r_norm_sq / p_Hp
            x_cg = x_cg + alpha * p
            r = r - alpha * Hp

            r_norm_sq_new = torch.sum(r ** 2)
            r_norm = torch.sqrt(r_norm_sq_new)
            num_iterations = i + 1

            if r_norm < cg_tol * r_norm_0:
                break

            if i < cg_max_iter - 1:
                beta = r_norm_sq_new / (r_norm_sq + 1e-10)
                p = r + beta * p
                r_norm_sq = r_norm_sq_new

        final_residual_ratio = r_norm / (r_norm_0 + 1e-10)
        return x_cg, num_iterations, final_residual_ratio.item()


def lanczos_eigenvalue_estimation(
    model,
    x_sample,
    t_timestep: int,
    k: int = 10,
    num_vectors: int = 5,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate maximum and minimum eigenvalues using Lanczos algorithm with multiple random vectors.

    Args:
        model: Model for Hessian computation
        x: Current sample tensor
        k: Number of Lanczos iterations
        num_vectors: Number of random vectors for robust estimation
        device: Device to use

    Returns:
        alpha_t: Maximum eigenvalue estimate (λ_max)
        beta_t: Minimum eigenvalue estimate (λ_min)
    """
    x_shape = x_sample.shape
    batch_size = x_shape[0]
    total_dim = int(np.prod(x_shape[1:]))

    max_eigenvals = []
    min_eigenvals = []

    # Use a representative sample from the batch (first element) for eigenvalue estimation
    # This is more efficient and the eigenvalues are typically similar across batch elements
    for vec_idx in range(num_vectors):
        # Initialize random vector (flattened)
        v_flat = torch.randn(total_dim, device=device, dtype=torch.float32)
        v_flat = v_flat / torch.norm(v_flat)
        v = v_flat.view(x_shape[0:1] + x_shape[1:])  # Expand to batch shape

        # Lanczos algorithm
        q = [v_flat.cpu().numpy()]
        alpha = []
        beta = []

        # First iteration
        v_full = v_flat.view(x_shape[0:1] + x_shape[1:])
        w = hessian_vector_product(model, x_sample, t_timestep, v_full)
        w_flat = w.view(batch_size, -1).mean(dim=0).cpu().numpy()  # Average over batch

        alpha_0 = np.dot(w_flat, v_flat.cpu().numpy())
        alpha.append(alpha_0)
        w_flat = w_flat - alpha_0 * v_flat.cpu().numpy()

        beta_0 = np.linalg.norm(w_flat)
        beta.append(beta_0)

        if beta_0 > 1e-8:
            q.append(w_flat / (beta_0 + 1e-8))
        else:
            # Early termination if breakdown
            q.append(v_flat.cpu().numpy())

        # Additional iterations
        for i in range(1, min(k, total_dim)):
            v_prev = q[-2]
            v_curr = q[-1]

            # Expand to full shape for HVP
            v_curr_full = torch.tensor(v_curr, device=device, dtype=torch.float32).view(x_shape[0:1] + x_shape[1:])
            w = hessian_vector_product(model, x_sample, t_timestep, v_curr_full)
            w_flat = w.view(batch_size, -1).mean(dim=0).cpu().numpy()

            alpha_i = np.dot(w_flat, v_curr)
            alpha.append(alpha_i)

            w_flat = w_flat - alpha_i * v_curr - beta[-1] * v_prev

            beta_i = np.linalg.norm(w_flat)
            beta.append(beta_i)

            if beta_i > 1e-8:
                q.append(w_flat / (beta_i + 1e-8))
            else:
                break

        # Build tridiagonal matrix and compute eigenvalues
        n = len(alpha)
        T = np.zeros((n, n))
        for i in range(n):
            T[i, i] = alpha[i]
            if i < n - 1 and len(beta) > i:
                T[i, i+1] = beta[i]
                T[i+1, i] = beta[i]

        # Compute eigenvalues
        eigenvals = np.linalg.eigvalsh(T)
        eigenvals = np.sort(eigenvals)[::-1]  # Descending order

        if len(eigenvals) > 0:
            max_eigenvals.append(float(eigenvals[0]))
            min_eigenvals.append(float(eigenvals[-1]))

    if len(max_eigenvals) == 0:
        # Fallback values
        alpha_t = torch.tensor(1.0, dtype=torch.float32, device=device)
        beta_t = torch.tensor(0.1, dtype=torch.float32, device=device)
    else:
        # Return robust estimates (mean across multiple vectors)
        alpha_t = torch.tensor(np.mean(max_eigenvals), dtype=torch.float32, device=device)
        beta_t = torch.tensor(np.mean(min_eigenvals), dtype=torch.float32, device=device)

    # Ensure positive definiteness with improved lower bound for numerical stability
    # Higher beta_min improves condition number control and numerical stability
    # Previously: 1e-6, now: 1e-5 (10x larger) to prevent extremely large condition numbers
    beta_min = 1e-5  # Increased from 1e-6 for better numerical stability
    beta_t = torch.clamp(beta_t, min=beta_min)
    alpha_t = torch.clamp(alpha_t, min=beta_t.item())

    return alpha_t, beta_t


def adaptive_damping_lambda(
    alpha_t: torch.Tensor,
    beta_t: torch.Tensor,
    kappa_target: float = 10.0
) -> torch.Tensor:
    f"""
    Compute adaptive damping parameter lambda_t based on condition number.

    According to the lemma:
    lambda_t = max(0, (alpha_t - kappa_* * beta_t) / (kappa_* - 1))

    This ensures kappa(H_sym + lambda_t I) ≤ kappa_*

    Note: lambda_scale is applied outside this function in hcg_correct()

    Args:
        alpha_t: Maximum eigenvalue (lambda_max)
        beta_t: Minimum eigenvalue (lambda_min)
        kappa_target: Target condition number \kappa_* (>1)

    Returns:
        lambda_t: Adaptive damping parameter (base value, unscaled)
    """
    if kappa_target <= 1.0:
        raise ValueError(f"kappa_target must be > 1, got {kappa_target}")

    # Current condition number
    kappa_current = alpha_t / (beta_t + 1e-8)

    if kappa_current <= kappa_target:
        # No damping needed
        return torch.tensor(0.0, dtype=alpha_t.dtype, device=alpha_t.device)

    # Compute adaptive damping
    numerator = alpha_t - kappa_target * beta_t
    denominator = kappa_target - 1.0

    lambda_t = torch.maximum(
        torch.tensor(0.0, dtype=alpha_t.dtype, device=alpha_t.device),
        numerator / denominator
    )

    return lambda_t


@profile_function("hcg_correct")
def hcg_correct(
    noise_pred: torch.Tensor,
    model,
    x_sample: torch.Tensor,
    t_timestep: int,
    device: str = 'cuda',
    kappa_target: float = 10.0,
    lanczos_k: int = 10,
    cg_max_iter: int = 20,
    cg_tol: float = 1e-4,
    use_spectral_radius: bool = True,
    spectral_scaling: float = 1.0,
    use_adaptive_lambda: bool = True,
    lambda_base: float = 0.004,
    lambda_scale: float = 1.0,
    log_lambda_stats: bool = False,
    cached_eigenvalues: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # New: optional cached eigenvalues
    prev_cg_solution: Optional[torch.Tensor] = None,  # New: previous CG solution for warm start
    # Additional control variables for debugging
    use_cg_warm_start: bool = True,  # Control whether to use CG warm start
    use_normalization: bool = True,  # Control whether to normalize after correction
    use_ema_smoothing: bool = False,  # Control whether to use EMA smoothing (like LML)
    ema_kappa: float = 1e-8,  # EMA smoothing factor (like LML's kappa)
    prev_noise: Optional[torch.Tensor] = None,  # Previous noise for EMA (like LML)
    skip_lanczos: bool = False,  # Skip Lanczos, use fixed eigenvalues for fast testing
    fixed_alpha: float = 1.0,  # Fixed alpha_t when skip_lanczos=True
    fixed_beta: float = 0.1,  # Fixed beta_t when skip_lanczos=True
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, dict]:
    """
    Hessian-Conjugate Gradient correction using adaptive damping.

    Solves c_t * (H + lambda_t I)^{-1} * noise_pred where:
    - H is the Hessian of -log p_t(x)
    - lambda_t = lambda_scale * adaptive_lambda is scaled adaptive damping
    - c_t = 1/(alpha_t + lambda_t) is the spectral radius scaling factor

    Args:
        noise_pred: Noise prediction to correct
        model: Model for Hessian computation
        x: Current sample
        t: Current timestep
        device: Device to use
        kappa_target: Target condition number κ_*
        lanczos_k: Number of Lanczos iterations for eigenvalue estimation
        cg_max_iter: Maximum CG iterations
        cg_tol: CG tolerance
        use_spectral_radius: Whether to apply spectral radius scaling c_t
        spectral_scaling: Spectral radius scaling factor c_t
        use_adaptive_lambda: Whether to use adaptive lambda_t
        lambda_base: Base lambda_t
        lambda_scale: Scaling factor for adaptive lambda_t (default: 1.0)
                     Use this to fine-tune the damping strength.
                     Typical range: 0.1 - 10.0 (similar to LML's fixed lambda ~0.0001-0.01)
        log_lambda_stats: Whether to log lambda_t statistics for debugging

    Returns:
        corrected_noise: Corrected noise prediction
    """
    batch_size, channels, height, width = noise_pred.shape

    # Step 1: Estimate eigenvalues using Lanczos (with caching if enabled)
    beta_min = 1e-5  # Minimum eigenvalue lower bound for numerical stability

    if skip_lanczos:
        # For debugging: use fixed eigenvalues
        alpha_t = torch.tensor(fixed_alpha, dtype=torch.float32, device=device)
        beta_t = torch.tensor(fixed_beta, dtype=torch.float32, device=device)
    elif cached_eigenvalues is not None:
        # Use cached eigenvalues if provided, but ensure they meet minimum bounds
        alpha_t, beta_t = cached_eigenvalues
        # Ensure cached values also respect the minimum bound
        beta_t = torch.clamp(beta_t, min=beta_min)
        alpha_t = torch.clamp(alpha_t, min=beta_t.item())
    else:
        try:
            with profile("lanczos_eigenvalue_estimation", get_profiler()):
                # Reduce num_vectors for efficiency (from 3 to 1-2)
                num_vectors = 2 if lanczos_k >= 5 else 1  # Fewer vectors when k is small
                alpha_t, beta_t = lanczos_eigenvalue_estimation(
                    model=model,
                    x_sample=x_sample,
                    t_timestep=t_timestep,
                    k=lanczos_k,
                    num_vectors=num_vectors,
                    device=device
                )
        except Exception as e:
            # Fallback: use diagonal approximation
            # This is a simplified fallback - in practice you might want better handling
            print(f"Warning: Lanczos estimation failed, using fallback: {e}")
            alpha_t = torch.tensor(1.0, dtype=torch.float32, device=device)
            beta_t = torch.tensor(max(0.1, beta_min), dtype=torch.float32, device=device)  # Ensure beta_t >= beta_min

    # Optional: Apply EMA smoothing (like LML)
    if use_ema_smoothing and prev_noise is not None:
        noise_pred_ema = ema_kappa * prev_noise + (1 - ema_kappa) * noise_pred
    else:
        noise_pred_ema = noise_pred

    # Step 2: Compute adaptive damping λ_t (simple computation, no profiling needed)
    # Step 2: Compute adaptive damping λ_t
    with profile("adaptive_damping_compute", get_profiler()):
        lambda_t_base = 0.0
    if use_adaptive_lambda:
        lambda_t_base = adaptive_damping_lambda(alpha_t, beta_t, kappa_target)
        lambda_t = lambda_scale * lambda_t_base
    else:
        lambda_t = lambda_base

    # Compute current condition numbers for analysis
    alpha_t_val = alpha_t.item() if isinstance(alpha_t, torch.Tensor) else alpha_t
    beta_t_val = beta_t.item() if isinstance(beta_t, torch.Tensor) else beta_t
    lambda_t_val = lambda_t.item() if isinstance(lambda_t, torch.Tensor) else lambda_t

    # κ(H_sym) = α_t / β_t (original condition number)
    kappa_original = alpha_t_val / (beta_t_val + 1e-8)

    # κ(A_t) = (α_t + λ_t) / (β_t + λ_t) (regularized condition number, per ICLR doc)
    # This is the condition number that actually affects CG convergence
    kappa_regularized = (alpha_t_val + lambda_t_val) / (beta_t_val + lambda_t_val + 1e-8)

    # Log statistics if requested (for understanding lambda_t range)
    if log_lambda_stats:
        lambda_t_item = lambda_t.item() if isinstance(lambda_t, torch.Tensor) else lambda_t
        print(f"[HCG lambda stats] t={t_timestep}, alpha_t={alpha_t_val:.6f}, beta_t={beta_t_val:.6f}, "
              f"kappa_original={kappa_original:.2f}, kappa_regularized={kappa_regularized:.2f}, "
              f"lambda_t={lambda_t_item:.6f}, "
              f"use_adaptive_lambda={use_adaptive_lambda}, lambda_base={lambda_base:.4f}, lambda_scale={lambda_scale:.4f}")

    # Step 3: Compute spectral radius scaling factor c_t = β_t + λ_t (per ICLR doc Lemma)
    # This ensures spec(M_t) ⊂ [1/κ_*, 1] where M_t(x) = c_t (H_sym(x) + λ_t I)^{-1}
    #
    # Theoretical basis:
    # - M_t = c_t * (H_sym + λ_t I)^{-1}
    # - With c_t = β_t + λ_t, we have:
    #   - λ_max(M_t) = c_t / (β_t + λ_t) = 1
    #   - λ_min(M_t) = c_t / (α_t + λ_t) = (β_t + λ_t) / (α_t + λ_t) = 1/κ(A_t) ≥ 1/κ_*
    # - This guarantees spec(M_t) ⊂ [1/κ_*, 1] as required by the Lemma
    #
    # Note: Using c_t = 1/(α_t + λ_t) would NOT satisfy this guarantee.
    # (Simple computation, no profiling needed)
    with profile("spectral_scaling_compute", get_profiler()):
        if use_spectral_radius:
            c_t = beta_t + lambda_t  # Per ICLR doc Lemma: ensures spec(M_t) ⊂ [1/κ_*, 1]
        else:
            c_t = spectral_scaling

    # Solve (H + λ_t I)^{-1} * noise_pred (with warm start if available)
    # Use noise_pred_ema if EMA smoothing is enabled
    cg_input = noise_pred_ema if use_ema_smoothing else noise_pred
    cg_prev_solution = prev_cg_solution if use_cg_warm_start else None
    corrected_noise, cg_iterations, cg_final_residual = conjugate_gradient_solve(
        model=model, x_sample=x_sample, t_timestep=t_timestep, prev_solution=cg_prev_solution, lambda_t=lambda_t, b=cg_input, cg_max_iter=cg_max_iter, cg_tol=cg_tol
    )

    # Step 5: Apply spectral radius scaling: c_t * corrected_noise
    with profile("spectral_scaling_apply", get_profiler()):
        corrected_noise = c_t * corrected_noise

    # Step 6: Normalize to preserve magnitude (optional, controlled by use_normalization)
    if use_normalization:
        with profile("hcg_normalization", get_profiler()):
            norm_original = torch.sqrt((noise_pred * noise_pred).sum(dim=(1, 2, 3), keepdim=True))
        norm_corrected = torch.sqrt((corrected_noise * corrected_noise).sum(dim=(1, 2, 3), keepdim=True))
        corrected_noise = corrected_noise * norm_original / (norm_corrected + 1e-8)

    # Compute theoretical minimum CG iterations k_pred (per ICLR doc convergence bound)
    # k_pred = 0.5 * sqrt(κ(A_t)) * log(2/τ)
    # This provides a theoretical lower bound for comparison with actual iterations k_obs
    theoretical_min_k = 0.5 * np.sqrt(kappa_regularized) * np.log(2.0 / cg_tol)

    # Prepare statistics for return (if statistics collection is needed)
    stats_dict = {
        'alpha_t': alpha_t_val,
        'beta_t': beta_t_val,
        'lambda_t_base': lambda_t_base.item() if isinstance(lambda_t_base, torch.Tensor) else lambda_t_base,
        'lambda_t': lambda_t_val,
        'kappa_original': kappa_original,  # κ(H_sym) = α_t / β_t
        'kappa_regularized': kappa_regularized,  # κ(A_t) = (α_t + λ_t) / (β_t + λ_t)
        'kappa_target': kappa_target,
        'c_t': c_t.item() if isinstance(c_t, torch.Tensor) else c_t,
        'cg_iterations': cg_iterations,  # k_obs: actual CG iterations
        'cg_final_residual': cg_final_residual,  # ||r^(k)|| / ||r^(0)||
        'cg_converged': cg_final_residual < cg_tol,
        'theoretical_min_k': theoretical_min_k,  # k_pred: theoretical lower bound
    }

    # Return corrected noise, eigenvalues (for caching), CG solution (for warm start), and stats
    return corrected_noise, (alpha_t, beta_t), corrected_noise.clone(), stats_dict


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999, alpha_transform_type="cosine"):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DPMSolverMultistepHCGScheduler(SchedulerMixin, ConfigMixin):
    """
    DPM-Solver with Hessian-Conjugate Gradient (HCG) correction using adaptive damping.

    This scheduler extends DPM-Solver with adaptive Hessian correction that:
    1. Estimates Hessian eigenvalues using Lanczos algorithm
    2. Computes adaptive damping λ_t based on target condition number
    3. Uses conjugate gradient to solve (H + λ_t I)^{-1} * g
    4. Applies spectral radius scaling c_t = 1/(α_t + λ_t)

    For more details, see the original DPM-Solver paper: https://arxiv.org/abs/2206.00927
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        use_karras_sigmas: Optional[bool] = False,
        lambda_min_clipped: float = -float("inf"),
        variance_type: Optional[str] = None,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        use_hcg: bool = True,
        kappa_target: float = 20.0,  # Increased from 10.0 for better performance
        lanczos_k: int = 5,  # Reduced from 10 for faster computation
        cg_max_iter: int = 5,  # Reduced from 20 for faster computation
        cg_tol: float = 1e-3,  # Balanced: tighter than 1e-2 for better convergence, but not as strict as 1e-4
        use_spectral_radius: bool = True,
        spectral_scaling: float = 1.0,
        use_adaptive_lambda: bool = True,
        lambda_base: float = 0.004,
        lambda_scale: float = 0.3,  # Reduced from 1.0 (typical range: 0.1-0.5)
        log_lambda_stats: bool = False,
        enable_eigenvalue_cache: bool = True,  # New: enable eigenvalue caching
        eigenvalue_cache_interval: int = 4,  # New: re-estimate every N steps (ICLR doc recommends r=2 or 4)
        # Additional control variables for debugging
        use_cg_warm_start: bool = True,  # Control whether to use CG warm start
        use_normalization: bool = True,  # Control whether to normalize after correction
        use_ema_smoothing: bool = False,  # Control whether to use EMA smoothing (like LML)
        ema_kappa: float = 1e-8,  # EMA smoothing factor (like LML's kappa)
        skip_lanczos: bool = False,  # Skip Lanczos, use fixed eigenvalues for fast testing
        fixed_alpha: float = 1.0,  # Fixed alpha_t when skip_lanczos=True
        fixed_beta: float = 0.1,  # Fixed beta_t when skip_lanczos=True
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.use_hcg = use_hcg
        self.kappa_target = kappa_target
        self.lanczos_k = lanczos_k
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.use_spectral_radius = use_spectral_radius
        self.spectral_scaling = spectral_scaling
        self.use_adaptive_lambda = use_adaptive_lambda
        self.lambda_base = lambda_base
        self.lambda_scale = lambda_scale
        self.log_lambda_stats = log_lambda_stats

        # Additional control variables for debugging
        self.use_cg_warm_start = use_cg_warm_start
        self.use_normalization = use_normalization
        self.use_ema_smoothing = use_ema_smoothing
        self.ema_kappa = ema_kappa
        self.skip_lanczos = skip_lanczos
        self.fixed_alpha = fixed_alpha
        self.fixed_beta = fixed_beta

        self.model = None  # Will be set during sampling
        self.prev_noise = None

        # Eigenvalue cache for optimization
        self.eigenvalue_cache = {}
        self.eigenvalue_cache_interval = eigenvalue_cache_interval
        self.enable_eigenvalue_cache = enable_eigenvalue_cache
        self.prev_cg_solution = None  # Cache for CG initial guess

        # Intermediate variables and statistics for analysis (as per paper/documentation)
        self.hcg_intermediate_vars = {
            'lambda_t_history': [],           # Adaptive damping λ_t sequence
            'lambda_t_base_history': [],       # Base lambda_t before scaling
            'kappa_original_history': [],      # Original condition number κ(H_sym) = α_t / β_t
            'kappa_regularized_history': [],   # Regularized condition number κ(A_t) = (α_t + λ_t) / (β_t + λ_t)
            'kappa_target_history': [],       # Target condition number (should be constant)
            'c_t_history': [],                # Spectral radius scaling factor sequence (c_t = β_t + λ_t per ICLR doc)
            'cg_iterations_history': [],       # CG iterations per step (k_obs)
            'cg_final_residual_history': [],   # CG final residual norms ||r^(k)|| / ||r^(0)||
            'theoretical_min_k_history': [],   # Theoretical lower bound k_pred = 0.5 * sqrt(κ(A_t)) * log(2/τ)
            'alpha_t_history': [],            # Maximum eigenvalue sequence
            'beta_t_history': [],              # Minimum eigenvalue sequence
            'timesteps_history': [],           # Corresponding timesteps
            'hvp_call_count': 0,               # Total HVP calls (for performance tracking)
            'eigenvalue_estimates_count': 0,   # Number of eigenvalue estimations
            'cg_convergence_history': [],      # CG convergence info (converged/not)
        }

        # Performance statistics
        self.hcg_performance_stats = {
            'lanczos_time_history': [],        # Time for Lanczos estimation
            'cg_time_history': [],             # Time for CG solve
            'hvp_time_history': [],            # Time for HVP computation
            'adaptive_damping_time_history': [], # Time for lambda_t computation
            'spectral_scaling_time_history': [], # Time for c_t computation
            'total_hcg_time_history': [],      # Total HCG correction time
        }

        # Enable/disable statistics collection (for performance)
        self.collect_stats = True  # Can be disabled if not needed

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Currently we only support VP-type noise schedule
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # settings for DPM-Solver
        if algorithm_type not in ["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"]:
            if algorithm_type == "deis":
                self.register_to_config(algorithm_type="dpmsolver++")
            else:
                raise NotImplementedError(f"{algorithm_type} does is not implemented for {self.__class__}")

        if solver_type not in ["midpoint", "heun"]:
            if solver_type in ["logrho", "bh1", "bh2"]:
                self.register_to_config(solver_type="midpoint")
            else:
                raise NotImplementedError(f"{solver_type} does is not implemented for {self.__class__}")

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0

    def set_model(self, model):
        """Set the model for Hessian computation"""
        self.model = model

    def _call_hcg_correct(self, noise, sample, timestep):
        """Helper method to call hcg_correct with caching and return only corrected noise"""
        # Get cached eigenvalues if available
        cached_eigs = None
        if self.enable_eigenvalue_cache and timestep in self.eigenvalue_cache:
            # Check if cache is still valid (within interval)
            cached_t, (alpha_cached, beta_cached) = self.eigenvalue_cache[timestep]
            if abs(timestep - cached_t) < self.eigenvalue_cache_interval:
                cached_eigs = (alpha_cached, beta_cached)

        corrected_noise, eigenvalues, cg_solution, stats = hcg_correct(
            noise_pred=noise,
            model=self.model,
            x_sample=sample,
            t_timestep=timestep,
            device=sample.device,
            kappa_target=self.kappa_target,
            lanczos_k=self.lanczos_k,
            cg_max_iter=self.cg_max_iter,
            cg_tol=self.cg_tol,
            use_spectral_radius=self.use_spectral_radius,
            spectral_scaling=self.spectral_scaling,
            use_adaptive_lambda=self.use_adaptive_lambda,
            lambda_base=self.lambda_base,
            lambda_scale=self.lambda_scale,
            log_lambda_stats=self.log_lambda_stats,
            cached_eigenvalues=cached_eigs,
            prev_cg_solution=self.prev_cg_solution,
            # Additional control variables
            use_cg_warm_start=getattr(self, 'use_cg_warm_start', True),
            use_normalization=getattr(self, 'use_normalization', True),
            use_ema_smoothing=getattr(self, 'use_ema_smoothing', False),
            ema_kappa=getattr(self, 'ema_kappa', 1e-8),
            prev_noise=self.prev_noise,
            skip_lanczos=getattr(self, 'skip_lanczos', False),
            fixed_alpha=getattr(self, 'fixed_alpha', 1.0),
            fixed_beta=getattr(self, 'fixed_beta', 0.1),
        )

        # Cache eigenvalues and CG solution for next step
        if self.enable_eigenvalue_cache:
            self.eigenvalue_cache[timestep] = (timestep, eigenvalues)
        self.prev_cg_solution = cg_solution

        # Save intermediate variables and statistics (if enabled)
        if hasattr(self, 'collect_stats') and self.collect_stats:
            self._save_hcg_statistics(timestep, stats)

        return corrected_noise

    def _save_hcg_statistics(self, timestep: int, stats: dict):
        """Save HCG statistics for analysis"""
        if not hasattr(self, 'hcg_intermediate_vars'):
            return

        self.hcg_intermediate_vars['timesteps_history'].append(timestep)
        self.hcg_intermediate_vars['alpha_t_history'].append(stats['alpha_t'])
        self.hcg_intermediate_vars['beta_t_history'].append(stats['beta_t'])
        self.hcg_intermediate_vars['lambda_t_base_history'].append(stats['lambda_t_base'])
        self.hcg_intermediate_vars['lambda_t_history'].append(stats['lambda_t'])
        self.hcg_intermediate_vars['kappa_original_history'].append(stats['kappa_original'])
        self.hcg_intermediate_vars['kappa_regularized_history'].append(stats['kappa_regularized'])
        self.hcg_intermediate_vars['kappa_target_history'].append(stats['kappa_target'])
        self.hcg_intermediate_vars['c_t_history'].append(stats['c_t'])
        self.hcg_intermediate_vars['cg_iterations_history'].append(stats['cg_iterations'])  # k_obs
        self.hcg_intermediate_vars['cg_final_residual_history'].append(stats['cg_final_residual'])
        self.hcg_intermediate_vars['theoretical_min_k_history'].append(stats['theoretical_min_k'])  # k_pred
        self.hcg_intermediate_vars['cg_convergence_history'].append(stats['cg_converged'])

        # Update counters
        self.hcg_intermediate_vars['hvp_call_count'] += (
            stats.get('lanczos_hvp_calls', 0) + stats.get('cg_hvp_calls', 0)
        )
        if stats.get('eigenvalue_estimated', False):
            self.hcg_intermediate_vars['eigenvalue_estimates_count'] += 1

    def get_hcg_statistics(self) -> dict:
        """Get collected HCG statistics for analysis"""
        if not hasattr(self, 'hcg_intermediate_vars'):
            return {}
        return {
            'intermediate_vars': self.hcg_intermediate_vars.copy(),
            'performance_stats': getattr(self, 'hcg_performance_stats', {}).copy(),
        }

    def clear_hcg_statistics(self):
        """Clear collected statistics (e.g., before new sampling)"""
        if hasattr(self, 'hcg_intermediate_vars'):
            for key in self.hcg_intermediate_vars:
                if isinstance(self.hcg_intermediate_vars[key], list):
                    self.hcg_intermediate_vars[key].clear()
                elif isinstance(self.hcg_intermediate_vars[key], (int, float)):
                    self.hcg_intermediate_vars[key] = 0
        if hasattr(self, 'hcg_performance_stats'):
            for key in self.hcg_performance_stats:
                if isinstance(self.hcg_performance_stats[key], list):
                    self.hcg_performance_stats[key].clear()

    def set_timesteps(self, num_inference_steps: int = None, device: Union[str, torch.device] = None):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        # Clipping the minimum of all lambda(t) for numerical stability.
        # This is critical for cosine (squaredcos_cap_v2) noise schedule.
        clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), self.config.lambda_min_clipped)
        last_timestep = ((self.config.num_train_timesteps - clipped_idx).numpy()).item()

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, last_timestep - 1, num_inference_steps + 1).round()[::-1][:-1].copy().astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = last_timestep // (num_inference_steps + 1)
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.arange(last_timestep, 0, -step_ratio).round().copy().astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        if self.config.use_karras_sigmas:
            log_sigmas = np.log(sigmas)
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
            timesteps = np.flip(timesteps).copy().astype(np.int64)

        self.sigmas = torch.from_numpy(sigmas)

        # when num_inference_steps == num_train_timesteps, we can end up with
        # duplicates in timesteps.
        _, unique_indices = np.unique(timesteps, return_index=True)
        timesteps = timesteps[np.sort(unique_indices)]

        self.timesteps = torch.from_numpy(timesteps).to(device)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0
        self.prev_noise = None

        # Clear caches when timesteps are reset
        if hasattr(self, 'eigenvalue_cache'):
            self.eigenvalue_cache.clear()
        if hasattr(self, 'prev_cg_solution'):
            self.prev_cg_solution = None

        # Clear statistics when starting new sampling
        if hasattr(self, 'collect_stats') and self.collect_stats:
            self.clear_hcg_statistics()

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, height, width = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * height * width)

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]

        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, height, width)
        sample = sample.to(dtype)

        return sample

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t
    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(sigma)

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    def _convert_to_karras(self, in_sigmas: torch.FloatTensor, num_inference_steps) -> torch.FloatTensor:
        """Constructs the noise schedule of Karras et al. (2022)."""

        sigma_min: float = in_sigmas[-1].item()
        sigma_max: float = in_sigmas[0].item()

        rho = 7.0  # 7.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def convert_model_output(
        self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Convert the model output to the corresponding type that the algorithm (DPM-Solver / DPM-Solver++) needs.

        DPM-Solver is designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to
        discretize an integral of the data prediction model. So we need to first convert the model output to the
        corresponding type to match the algorithm.

        Note that the algorithm type and the model type is decoupled. That is to say, we can use either DPM-Solver or
        DPM-Solver++ for both noise prediction model and data prediction model.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the converted model output.
        """

        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.config.prediction_type == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                if self.config.variance_type in ["learned", "learned_range"]:
                    model_output = model_output[:, :3]
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred

        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.config.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            if self.config.prediction_type == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                if self.config.variance_type in ["learned", "learned_range"]:
                    epsilon = model_output[:, :3]
                else:
                    epsilon = model_output
            elif self.config.prediction_type == "sample":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = (sample - alpha_t * model_output) / sigma_t
            elif self.config.prediction_type == "v_prediction":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = (sample - sigma_t * epsilon) / alpha_t
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t

            return epsilon

    def dpm_solver_first_order_update(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        One step for the first-order DPM-Solver (equivalent to DDIM).

        See https://arxiv.org/abs/2206.00927 for the detailed derivation.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
        lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
        alpha_t, alpha_s = self.alpha_t[prev_timestep], self.alpha_t[timestep]
        sigma_t, sigma_s = self.sigma_t[prev_timestep], self.sigma_t[timestep]
        h = lambda_t - lambda_s

        if self.config.algorithm_type == "dpmsolver++":
            noise = - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
            if self.use_hcg and self.model is not None:
                corrected_noise = self._call_hcg_correct(noise, sample, timestep)
                x_t = (sigma_t / sigma_s) * sample + corrected_noise
            else:
                x_t = (sigma_t / sigma_s) * sample + noise
            self.prev_noise = noise
        elif self.config.algorithm_type == "dpmsolver":
            noise = - (sigma_t * (torch.exp(h) - 1.0)) * model_output
            if self.use_hcg and self.model is not None:
                corrected_noise = self._call_hcg_correct(noise, sample, timestep)
                x_t = (alpha_t / alpha_s) * sample + corrected_noise
            else:
                x_t = (alpha_t / alpha_s) * sample + noise
            self.prev_noise = noise
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                (alpha_t / alpha_s) * sample
                - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            )
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        One step for the second-order multistep DPM-Solver.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)

        if self.config.algorithm_type == "dpmsolver++":
            if self.config.solver_type == "midpoint":
                noise = - (alpha_t * (torch.exp(-h) - 1.0)) * D0 - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                if self.use_hcg and self.model is not None:
                    corrected_noise = self._call_hcg_correct(noise, sample, timestep_list[-1])
                    x_t = (sigma_t / sigma_s0) * sample + corrected_noise
                else:
                    x_t = (sigma_t / sigma_s0) * sample + noise
                self.prev_noise = noise
            elif self.config.solver_type == "heun":
                noise = - (alpha_t * (torch.exp(-h) - 1.0)) * D0 + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                if self.use_hcg and self.model is not None:
                    corrected_noise = self._call_hcg_correct(noise, sample, timestep_list[-1])
                    x_t = (sigma_t / sigma_s0) * sample + corrected_noise
                else:
                    x_t = (sigma_t / sigma_s0) * sample + noise
                self.prev_noise = noise
        elif self.config.algorithm_type == "dpmsolver":
            if self.config.solver_type == "midpoint":
                noise = - (sigma_t * (torch.exp(h) - 1.0)) * D0 - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                if self.use_hcg and self.model is not None:
                    corrected_noise = self._call_hcg_correct(noise, sample, timestep_list[-1])
                    x_t = (alpha_t / alpha_s0) * sample + corrected_noise
                else:
                    x_t = (alpha_t / alpha_s0) * sample + noise
                self.prev_noise = noise
            elif self.config.solver_type == "heun":
                noise = - (sigma_t * (torch.exp(h) - 1.0)) * D0 - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                if self.use_hcg and self.model is not None:
                    corrected_noise = self._call_hcg_correct(noise, sample, timestep_list[-1])
                    x_t = (alpha_t / alpha_s0) * sample + corrected_noise
                else:
                    x_t = (alpha_t / alpha_s0) * sample + noise
                self.prev_noise = noise
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * (torch.exp(h) - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
        return x_t

    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        One step for the third-order multistep DPM-Solver.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
        t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        lambda_t, lambda_s0, lambda_s1, lambda_s2 = (
            self.lambda_t[t],
            self.lambda_t[s0],
            self.lambda_t[s1],
            self.lambda_t[s2],
        )
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)

        if self.config.algorithm_type == "dpmsolver++":
            noise = - (alpha_t * (torch.exp(-h) - 1.0)) * D0 + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1 - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
            if self.use_hcg and self.model is not None:
                corrected_noise = self._call_hcg_correct(noise, sample, timestep_list[-1])
                x_t = (sigma_t / sigma_s0) * sample + corrected_noise
            else:
                x_t = (sigma_t / sigma_s0) * sample + noise
            self.prev_noise = noise
        elif self.config.algorithm_type == "dpmsolver":
            noise = - (sigma_t * (torch.exp(h) - 1.0)) * D0 - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1 - (sigma_t * ((torch.exp(h) - 1.0 - h) / h**2 - 0.5)) * D2
            if self.use_hcg and self.model is not None:
                corrected_noise = self._call_hcg_correct(noise, sample, timestep_list[-1])
                x_t = (alpha_t / alpha_s0) * sample + corrected_noise
            else:
                x_t = (alpha_t / alpha_s0) * sample + noise
            self.prev_noise = noise
        return x_t

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the multistep DPM-Solver.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = len(self.timesteps) - 1
        else:
            step_index = step_index.item()
        prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1]
        lower_order_final = (
            (step_index == len(self.timesteps) - 1) and self.config.lower_order_final and len(self.timesteps) < 15
        )
        lower_order_second = (
            (step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, timestep, sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        if self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
        else:
            noise = None

        # For HCG, we need gradients enabled even if called from no_grad context
        # Wrap the update in enable_grad context when HCG is used
        if self.use_hcg and self.model is not None:
            with torch.enable_grad():
                if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
                    prev_sample = self.dpm_solver_first_order_update(
                        model_output, timestep, prev_timestep, sample, noise=noise
                    )
                elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
                    timestep_list = [self.timesteps[step_index - 1], timestep]
                    prev_sample = self.multistep_dpm_solver_second_order_update(
                        self.model_outputs, timestep_list, prev_timestep, sample, noise=noise
                    )
                else:
                    timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1], timestep]
                    prev_sample = self.multistep_dpm_solver_third_order_update(
                        self.model_outputs, timestep_list, prev_timestep, sample
                    )
                # Detach the result to remove from computation graph
                prev_sample = prev_sample.detach()
        else:
            if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
                prev_sample = self.dpm_solver_first_order_update(
                    model_output, timestep, prev_timestep, sample, noise=noise
                )
            elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
                timestep_list = [self.timesteps[step_index - 1], timestep]
                prev_sample = self.multistep_dpm_solver_second_order_update(
                    self.model_outputs, timestep_list, prev_timestep, sample, noise=noise
                )
            else:
                timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1], timestep]
                prev_sample = self.multistep_dpm_solver_third_order_update(
                    self.model_outputs, timestep_list, prev_timestep, sample
                )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
