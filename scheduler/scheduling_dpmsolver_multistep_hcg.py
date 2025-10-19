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
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput

def lm_correct_advanced(prev_noise, noise_pred, lamb, kappa, hessian_method='original',
                       model=None, x=None, t=None, device='cuda'):
    """
    Advanced LML correction with different Hessian computation methods

    Args:
        prev_noise: Previous noise prediction
        noise_pred: Current noise prediction
        lamb: Regularization parameter
        kappa: EMA parameter
        hessian_method: 'original', 'explicit', or 'hessian_free'
        model: Model for Hessian computation (if needed)
        x: Current sample (if needed)
        t: Current timestep (if needed)
        device: Device to use
    """

    if prev_noise is not None:
        noise_pred_ema = kappa * prev_noise + (1 - kappa) * noise_pred
    else:
        noise_pred_ema = noise_pred

    if hessian_method == 'original':
        # Original LML method (simplified approximation)
        return lm_correct_original(noise_pred, noise_pred_ema, lamb)

    elif hessian_method == 'explicit':
        # Explicit Hessian computation
        return lm_correct_explicit_hessian(noise_pred, noise_pred_ema, lamb, model, x, t, device)

    elif hessian_method == 'hessian_free':
        # Hessian-Free method using CG + HVP
        return lm_correct_hessian_free(noise_pred, noise_pred_ema, lamb, model, x, t, device)

    else:
        raise ValueError(f"Unknown hessian_method: {hessian_method}")

def lm_correct_original(noise_pred, noise_pred_ema, lamb):
    """Original LML correction method"""
    # Original implementation
    norm_squared = (noise_pred * noise_pred).sum(dim=(1, 2, 3))
    norm_squared = norm_squared.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    part1 = noise_pred

    norm_squared_ema = (noise_pred_ema * noise_pred_ema).sum(dim=(1, 2, 3))
    norm_squared_ema = norm_squared_ema.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    inner_product = torch.sum(noise_pred * noise_pred_ema, dim=(1, 2, 3))
    mp = noise_pred_ema * inner_product.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    part2 = mp / (lamb + norm_squared_ema)

    inversed_pred = part1 - part2

    # normalize the direction
    norm = torch.sqrt(norm_squared)
    norm_squared_lm = (inversed_pred * inversed_pred).sum(dim=(1, 2, 3))
    norm_squared_lm = norm_squared_lm.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    norm_lm = torch.sqrt(norm_squared_lm)
    inversed_pred = inversed_pred * norm / norm_lm
    return inversed_pred

def lm_correct_explicit_hessian(noise_pred, noise_pred_ema, lamb, model, x, t, device):
    """
    LML correction using explicit Hessian computation
    Based on finite difference approximation of Hessian matrix
    """
    try:
        batch_size, channels, height, width = noise_pred.shape

        # Compute gradient of log p_t(x) at current point
        x_grad = x.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            # Forward pass to get score function
            score_pred = model(x_grad, t)
            if hasattr(score_pred, 'sample'):
                score_pred = score_pred.sample

            # Log probability (simplified)
            log_prob = -0.5 * torch.sum(score_pred ** 2, dim=(1, 2, 3))
            log_prob = log_prob.sum()

        # Compute gradient
        grad = torch.autograd.grad(log_prob, x_grad, create_graph=True)[0]

        # Compute Hessian using finite differences (simplified version)
        # For efficiency, we only compute diagonal elements
        eps = 1e-4
        hessian_diag = torch.zeros_like(x)

        for i in range(channels):
            for j in range(height):
                for k in range(width):
                    # Perturb single element
                    x_pert = x.clone()
                    x_pert[:, i, j, k] += eps

                    # Compute gradient at perturbed point
                    x_pert_grad = x_pert.clone().detach().requires_grad_(True)
                    with torch.enable_grad():
                        score_pred_pert = model(x_pert_grad, t)
                        if hasattr(score_pred_pert, 'sample'):
                            score_pred_pert = score_pred_pert.sample
                        log_prob_pert = -0.5 * torch.sum(score_pred_pert ** 2, dim=(1, 2, 3))
                        log_prob_pert = log_prob_pert.sum()

                    grad_pert = torch.autograd.grad(log_prob_pert, x_pert_grad, create_graph=False)[0]
                    hessian_diag[:, i, j, k] = (grad_pert[:, i, j, k] - grad[:, i, j, k]) / eps

        # Apply LML correction using diagonal Hessian approximation
        # H^{-1} ≈ (H_diag + λI)^{-1}
        hessian_inv_diag = 1.0 / (hessian_diag + lamb)

        # Apply correction
        corrected_noise = noise_pred * hessian_inv_diag

        # Normalize
        norm = torch.sqrt((noise_pred * noise_pred).sum(dim=(1, 2, 3), keepdim=True))
        norm_corrected = torch.sqrt((corrected_noise * corrected_noise).sum(dim=(1, 2, 3), keepdim=True))
        corrected_noise = corrected_noise * norm / (norm_corrected + 1e-8)

        return corrected_noise

    except Exception as e:
        print(f"Error in explicit Hessian computation: {e}")
        # Fallback to original method
        return lm_correct_original(noise_pred, noise_pred_ema, lamb)

def lm_correct_hessian_free(noise_pred, noise_pred_ema, lamb, model, x, t, device):
    """
    LML correction using Hessian-Free method (CG + HVP)
    Solves H^{-1}g using conjugate gradient without explicit Hessian
    """
    try:
        batch_size, channels, height, width = noise_pred.shape

        def hessian_vector_product(v):
            """Compute Hv using Pearlmutter's method"""
            # Ensure v requires grad
            v_grad = v.clone().detach().requires_grad_(True)
            x_grad = x.clone().detach().requires_grad_(True)

            with torch.enable_grad():
                score_pred = model(x_grad, t)
                if hasattr(score_pred, 'sample'):
                    score_pred = score_pred.sample
                log_prob = -0.5 * torch.sum(score_pred ** 2, dim=(1, 2, 3))
                log_prob = log_prob.sum()

            # First gradient
            grad = torch.autograd.grad(log_prob, x_grad, create_graph=True)[0]

            # Hv = ∇(∇f · v)
            grad_dot_v = torch.sum(grad * v_grad)
            hv = torch.autograd.grad(grad_dot_v, x_grad, retain_graph=True)[0]
            return hv

        def conjugate_gradient_solve(b, max_iter=20, tol=1e-4):
            """Solve Hx = b using CG"""
            x_cg = torch.zeros_like(b)
            r = b.clone()
            p = r.clone()

            r_norm_sq = torch.sum(r ** 2)
            r_norm_0 = torch.sqrt(r_norm_sq)

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

                if r_norm < tol * r_norm_0:
                    break

                beta = r_norm_sq_new / r_norm_sq
                p = r + beta * p
                r_norm_sq = r_norm_sq_new

            return x_cg

        # Solve H^{-1} * noise_pred using CG
        # We want to solve Hx = noise_pred, so x = H^{-1} * noise_pred
        corrected_noise = conjugate_gradient_solve(noise_pred)

        # Add regularization
        corrected_noise = corrected_noise / (1.0 + lamb)

        # Normalize
        norm = torch.sqrt((noise_pred * noise_pred).sum(dim=(1, 2, 3), keepdim=True))
        norm_corrected = torch.sqrt((corrected_noise * corrected_noise).sum(dim=(1, 2, 3), keepdim=True))
        corrected_noise = corrected_noise * norm / (norm_corrected + 1e-8)

        return corrected_noise

    except Exception as e:
        print(f"Error in Hessian-Free computation: {e}")
        # Fallback to original method
        return lm_correct_original(noise_pred, noise_pred_ema, lamb)

# Import the rest of the original scheduler
from scheduler.scheduling_dpmsolver_multistep_lm import (
    betas_for_alpha_bar, DPMSolverMultistepLMScheduler
)

class DPMSolverMultistepHCGScheduler(DPMSolverMultistepLMScheduler):
    """
    Advanced DPM-Solver with multiple Hessian computation methods
    """

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
        lamb: float = 1.0,
        lm: bool = False,
        kappa: float = 0.0,
        hessian_method: str = 'original',  # New parameter
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            solver_order=solver_order,
            prediction_type=prediction_type,
            thresholding=thresholding,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            sample_max_value=sample_max_value,
            algorithm_type=algorithm_type,
            solver_type=solver_type,
            lower_order_final=lower_order_final,
            use_karras_sigmas=use_karras_sigmas,
            lambda_min_clipped=lambda_min_clipped,
            variance_type=variance_type,
            timestep_spacing=timestep_spacing,
            steps_offset=steps_offset,
            lamb=lamb,
            lm=lm,
            kappa=kappa,
        )
        self.hessian_method = hessian_method
        self.model = None  # Will be set during sampling

    def set_model(self, model):
        """Set the model for Hessian computation"""
        self.model = model

    def dpm_solver_first_order_update(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
        lamb: float = 1.0,
        lm=True,
    ) -> torch.FloatTensor:
        """Enhanced first-order update with advanced Hessian methods"""
        lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
        alpha_t, alpha_s = self.alpha_t[prev_timestep], self.alpha_t[timestep]
        sigma_t, sigma_s = self.sigma_t[prev_timestep], self.sigma_t[timestep]
        h = lambda_t - lambda_s

        if self.config.algorithm_type == "dpmsolver++":
            noise = - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
            if lm is True:
                x_t = (sigma_t / sigma_s) * sample + lm_correct_advanced(
                    prev_noise=self.prev_noise,
                    noise_pred=noise,
                    lamb=self.lamb,
                    kappa=self.kappa,
                    hessian_method=self.hessian_method,
                    model=self.model,
                    x=sample,
                    t=timestep,
                    device=sample.device
                )
            else:
                x_t = (sigma_t / sigma_s) * sample + noise
            self.prev_noise = noise
        elif self.config.algorithm_type == "dpmsolver":
            noise = - (sigma_t * (torch.exp(h) - 1.0)) * model_output
            if lm is True:
                x_t = (alpha_t / alpha_s) * sample + lm_correct_advanced(
                    prev_noise=self.prev_noise,
                    noise_pred=noise,
                    lamb=self.lamb,
                    kappa=self.kappa,
                    hessian_method=self.hessian_method,
                    model=self.model,
                    x=sample,
                    t=timestep,
                    device=sample.device
                )
            else:
                x_t = (alpha_t / alpha_s) * sample + noise
            self.prev_noise = noise
        else:
            # Handle other algorithm types
            x_t = super().dpm_solver_first_order_update(
                model_output, timestep, prev_timestep, sample, noise, lamb, lm
            )

        return x_t
