#!/usr/bin/env python3
"""
PCA2 Analysis: XT Space PCA2, PC2/PC1 Ratio
Optimized version: Uses global PCA basis from 1000 seeds, then projects single seed trajectory
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import sys
import os
import argparse
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set matplotlib to use a backend that doesn't require display
import matplotlib
matplotlib.use('Agg')

# Import schedulers
sys.path.append(os.getcwd())
from diffusers import DDPMPipeline, LDMPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers import DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler
from scheduler.scheduling_pndm_hcg import PNDMSHCGcheduler
from scheduler.scheduling_unipc_multistep_hcg import UniPCMultistepHCGScheduler

import project as project

cifar10_model_path = os.path.join(project.model_dir, 'ddpm_ema_cifar10')
celeba_model_path = "/home/liying/Documents/ldm-celebahq-256/"
coco_sd2_model_path = "/home/liying/Documents/stable-diffusion-2-base"
coco_sd15_model_path = "/home/liying/Documents/stable-diffusion-v1-5"
coco_sdxl_model_path = "/home/liying/Documents/stable-diffusion-xl-base-1.0"

coco_prompts_path = os.path.join(project.project_dir, "evaluations", "coco_prompts", "coco_top_40_prompts.json")

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Set matplotlib parameters for ICLR paper format
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 3,
    'lines.markersize': 6,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3
})

class PCA2Analysis:
    """
    PCA2 Analysis class for analyzing diffusion sampling trajectories.

    This class performs Principal Component Analysis (PCA) on diffusion trajectories
    in the XT space (image state space). It first computes a global PCA basis from
    1000 seed trajectories, then projects a single seed trajectory onto this basis.

    Key analyses:
    - Global PCA Basis: Computed from 1000 seed trajectories across all inference steps
    - XT Space PCA2: 2D projection of single seed trajectory using global PCA basis
    - PC2/PC1 Ratio: Measures anisotropy at each inference step using projected weights
    """

    def __init__(self, method='ddim', model='ddpm_ema_cifar10', num_inference_steps=1000,
                 num_global_seeds=1000, seed=42):
        """
        Initialize PCA2 Analysis.

        Args:
            method: Sampling method (ddim, dpm, dpm_lm, unipc, etc.)
            model: Model type (ddpm_ema_cifar10, ldm_celebahq_256, stable-diffusion-2-base, etc.)
            num_inference_steps: Number of inference steps in the diffusion process
            num_global_seeds: Number of seeds to use for computing global PCA basis
            seed: Random seed for single trajectory analysis
        """
        self.method = method
        self.model = model
        self.num_inference_steps = num_inference_steps
        self.num_global_seeds = num_global_seeds
        self.seed = seed

        # Generate postfix for output file names
        self.pic_postfix_name = f'{self.model}_{self.method}_steps-{self.num_inference_steps}_seed-{self.seed}'
        self.global_pca_postfix = f'{self.model}_{self.method}_steps-{self.num_inference_steps}_global-{self.num_global_seeds}'

        print(f"🔧 PCA2 Analysis Configuration:")
        print(f"   - Method: {self.method}")
        print(f"   - Model: {self.model}")
        print(f"   - Inference steps per trajectory: {self.num_inference_steps}")
        print(f"   - Number of global seeds for PCA basis: {self.num_global_seeds}")
        print(f"   - Single trajectory seed: {self.seed}")

    def load_pipeline(self):
        print(f"\n🔧 Loading {self.method.upper()} {self.model.upper()} pipeline...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.model == 'ddpm_ema_cifar10':
            pipe = DDPMPipeline.from_pretrained(cifar10_model_path, torch_dtype=torch.float32, use_safetensors=False)
            pipe.unet.to(device)
            self.setup_cifar10_scheduler(pipe, self.method)

        elif self.model == 'ldm_celebahq_256':
            pipe = LDMPipeline.from_pretrained(celeba_model_path, torch_dtype=torch.float32, use_safetensors=False)
            pipe.unet.to(device)
            pipe.vqvae.to(device)
            pipe.vqvae.config.scaling_factor = 1.0
            self.setup_celeba_scheduler(pipe, self.method)

        elif self.model == 'stable-diffusion-2-base':
            pipe = StableDiffusionPipeline.from_pretrained(coco_sd2_model_path, torch_dtype=torch.float32, use_safetensors=False)
            pipe.unet.to(device)
            self.setup_sd_scheduler(pipe, self.method)

        elif self.model == 'stable-diffusion-xl-base-1.0':
            pipe = StableDiffusionXLPipeline.from_pretrained(coco_sdxl_model_path, torch_dtype=torch.float32, use_safetensors=False, safety_checker=None, added_cond_kwargs={})
            pipe.unet.to(device)
            self.setup_sd_scheduler(pipe, self.method)

        elif self.model == 'stable-diffusion-v1-5':
            pipe = StableDiffusionPipeline.from_pretrained(coco_sd15_model_path, torch_dtype=torch.float32, use_safetensors=False)
            pipe.unet.to(device)
            self.setup_sd_scheduler(pipe, self.method)

        else:
            raise ValueError(f"Unknown model type: {self.model}")

        pipe = pipe.to(device)

        # Setup DDIM scheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(self.num_inference_steps)

        print(f"✓ {self.method.upper()} pipeline loaded successfully")
        return pipe

    def setup_cifar10_scheduler(self, pipe, sampler_type, lamb=0.0008, kappa=1e-8):
        """Setup the appropriate scheduler based on sampler type"""
        if sampler_type == 'pndm':
            pipe.scheduler = PNDMSHCGcheduler.from_config(pipe.scheduler.config)
            print(f"  Using PNDM scheduler")
        elif sampler_type == 'ddim':
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            print(f"  Using DDIM scheduler")
        elif sampler_type == 'dpm++':
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver++"
            print(f"  Using DPM-Solver++ scheduler")
        elif sampler_type == 'dpm_lm':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lamb = lamb
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = kappa
            print(f"  Using DPM-Solver with LML correction (λ={lamb}, κ={kappa})")
        elif sampler_type == 'dpm':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lm = False
            print(f"  Using DPM-Solver scheduler")
        elif sampler_type == 'unipc':
            pipe.scheduler = UniPCMultistepHCGScheduler.from_config(pipe.scheduler.config)
            print(f"  Using UniPC scheduler")
        elif sampler_type == 'dpm_hcg':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver++"
            pipe.scheduler.lamb = lamb
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = kappa
            print(f"  Using DPM-Solver with LML correction (λ={lamb}, κ={kappa})")
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

    def setup_celeba_scheduler(self, pipe, sampler_type, lamb=0.0008, kappa=1e-8):
        """Setup the appropriate scheduler based on sampler type"""
        original_config = pipe.scheduler.config
        config_dict = {
            k: v for k, v in original_config.items()
            if k not in ['timestep_values', 'timesteps']
        }

        if sampler_type == 'pndm':
            pipe.scheduler = PNDMScheduler.from_config(config_dict)
            project.info(f"  Using PNDM scheduler")
        elif sampler_type == 'ddim':
            pipe.scheduler = DDIMScheduler.from_config(config_dict)
            pipe.scheduler.config.eta = 0.0
            project.info(f"  Using DDIM scheduler (eta=0.0)")
        elif sampler_type == 'ddim_lm':
            pipe.scheduler = DDIMLMScheduler.from_config(config_dict)
            pipe.scheduler.lamb = lamb
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = kappa
            project.info(f"  Using DDIM with LML correction (λ={lamb}, κ={kappa})")
        elif sampler_type == 'dpm++':
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(config_dict)
            pipe.scheduler.config.algorithm_type = "dpmsolver++"
            pipe.scheduler.config.solver_order = 3
            project.info(f"  Using DPM-Solver++ scheduler")
        elif sampler_type == 'dpm_lm':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(config_dict)
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.lamb = lamb
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = kappa
            project.info(f"  Using DPM-Solver with LML correction (λ={lamb}, κ={kappa})")
        elif sampler_type == 'dpm':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(config_dict)
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.lm = False
            project.info(f"  Using DPM-Solver scheduler")
        elif sampler_type == 'unipc':
            pipe.scheduler = UniPCMultistepScheduler.from_config(config_dict)
            project.info(f"  Using UniPC scheduler")
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

    def setup_sd_scheduler(self, pipe, sampler_type, lamb=5.0, kappa=0.0):
        """Setup the appropriate scheduler based on sampler type"""
        if sampler_type == 'pndm':
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
            print(f"  Using PNDM scheduler")
        elif sampler_type == 'ddim':
            pipe.scheduler = DDIMLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.lamb = lamb
            pipe.scheduler.lm = False
            pipe.scheduler.kappa = kappa
            print(f"  Using DDIM scheduler")
        elif sampler_type == 'ddim_lm':
            pipe.scheduler = DDIMLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.lamb = lamb
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = kappa
            print(f"  Using DDIM with LML correction (λ={lamb}, κ={kappa})")
        elif sampler_type == 'dpm++':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver++"
            pipe.scheduler.lamb = lamb
            pipe.scheduler.lm = False
            print(f"  Using DPM-Solver++ scheduler")
        elif sampler_type == 'dpm_lm':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lamb = lamb
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = kappa
            print(f"  Using DPM-Solver with LML correction (λ={lamb}, κ={kappa})")
        elif sampler_type == 'dpm':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lamb = lamb
            pipe.scheduler.lm = False
            print(f"  Using DPM-Solver scheduler")
        elif sampler_type == 'unipc':
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            print(f"  Using UniPC scheduler")
        elif sampler_type == 'dpm_hcg':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver++"
            pipe.scheduler.lamb = lamb
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = kappa
            print(f"  Using DPM-Solver with LML correction (λ={lamb}, κ={kappa})")
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

    def load_coco_prompts(self, coco_prompts_path=coco_prompts_path):
        """Load COCO prompts from JSON file"""
        try:
            import json
            with open(coco_prompts_path) as fr:
                COCO_prompts_dict = json.load(fr)
            return COCO_prompts_dict
        except FileNotFoundError:
            print("⚠️  COCO prompts file not found. Using default prompts.")
            return {
                "00001": "a beautiful landscape with mountains and trees",
                "00002": "a cat sitting on a windowsill",
                "00003": "a modern city skyline at sunset",
                "00004": "a vintage car parked on the street",
                "00005": "a delicious plate of pasta"
            }

    def _generate_single_trajectory(self, pipe, seed, model_type="ddpm_ema_cifar10"):
        """
        Generate a single trajectory based on model type.

        Args:
            pipe: Pipeline instance
            seed: Random seed for trajectory generation
            model_type: Type of model

        Returns:
            trajectory_data: Dictionary containing 'xt', 'timesteps'
        """
        torch.manual_seed(seed)

        coco_prompts_dict = self.load_coco_prompts()
        if seed in coco_prompts_dict:
            prompt = coco_prompts_dict[seed]
        else:
            prompt = "a beautiful landscape with mountains and trees"

        if model_type == 'ddpm_ema_cifar10':
            trajectory_data = self._generate_single_trajectory_cifar10(pipe, seed)
        elif model_type == 'ldm_celebahq_256':
            trajectory_data = self._generate_single_trajectory_celeba(pipe, seed)
        elif model_type in ['stable-diffusion-2-base', 'stable-diffusion-xl-base-1.0', 'stable-diffusion-v1-5']:
            trajectory_data = self._generate_single_trajectory_sd(pipe, seed, prompt)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return trajectory_data

    def _generate_single_trajectory_cifar10(self, pipe, seed):
        """Generate a single trajectory and collect intermediate states"""
        torch.manual_seed(seed)

        device = pipe.unet.device
        shape = (1, pipe.unet.config.in_channels, 32, 32)
        latents = torch.randn(shape, device=device)

        trajectory_data = {
            'xt': [],
            'timesteps': []
        }

        scheduler = pipe.scheduler

        for j, t in enumerate(scheduler.timesteps):
            xt_flat = latents.detach().view(1, -1).cpu().numpy().flatten()
            trajectory_data['xt'].append(xt_flat)
            trajectory_data['timesteps'].append(t.item())

            with torch.no_grad():
                noise_pred = pipe.unet(latents, t).sample

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        for key in ['xt', 'timesteps']:
            trajectory_data[key] = np.array(trajectory_data[key])

        return trajectory_data

    def _generate_single_trajectory_celeba(self, pipe, seed):
        """Generate a single trajectory for CelebA pipeline"""
        torch.manual_seed(seed)
        device = pipe.unet.device

        height = pipe.vqvae.config.sample_size if hasattr(pipe.vqvae.config, 'sample_size') else 256
        width = height
        latent_channels = pipe.unet.config.in_channels
        shape = (1, latent_channels, height // 8, width // 8)
        latents = torch.randn(shape, device=device, dtype=pipe.unet.dtype)

        pipe.scheduler.set_timesteps(self.num_inference_steps, device=device)

        trajectory_data = {
            'xt': [],
            'timesteps': []
        }

        scheduler = pipe.scheduler

        for j, t in enumerate(scheduler.timesteps):
            xt_flat = latents.detach().view(1, -1).cpu().numpy().flatten()
            trajectory_data['xt'].append(xt_flat)
            trajectory_data['timesteps'].append(t.item() if isinstance(t, torch.Tensor) else t)

            with torch.no_grad():
                noise_pred = pipe.unet(latents, t).sample

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        for key in ['xt', 'timesteps']:
            trajectory_data[key] = np.array(trajectory_data[key])
        return trajectory_data

    def _generate_single_trajectory_sd(self, pipe, seed, prompt="a beautiful landscape with mountains and trees",
                                       negative_prompt="", guidance_scale=7.5):
        """Generate a single trajectory for Stable Diffusion pipeline"""
        torch.manual_seed(seed)
        device = pipe.device

        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

        if negative_prompt is None:
            negative_prompt = ""
        uncond_tokens = pipe.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = pipe.text_encoder(uncond_tokens.input_ids.to(device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        vae_scale_factor = getattr(pipe, 'vae_scale_factor', 8)
        if hasattr(pipe.unet.config, 'sample_size'):
            height = width = pipe.unet.config.sample_size * vae_scale_factor
        else:
            height = width = 512 if 'xl' not in pipe.__class__.__name__.lower() else 1024
        latent_channels = pipe.unet.config.in_channels
        shape = (1, latent_channels, height // vae_scale_factor, width // vae_scale_factor)
        latents = torch.randn(shape, device=device, dtype=text_embeddings.dtype)

        pipe.scheduler.set_timesteps(self.num_inference_steps, device=device)
        latents = latents * pipe.scheduler.init_noise_sigma

        trajectory_data = {
            'xt': [],
            'timesteps': []
        }

        scheduler = pipe.scheduler

        for j, t in enumerate(scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            xt_flat = latents.detach().view(1, -1).cpu().numpy().flatten()
            trajectory_data['xt'].append(xt_flat)
            trajectory_data['timesteps'].append(t.item() if isinstance(t, torch.Tensor) else t)

            with torch.no_grad():
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        for key in ['xt', 'timesteps']:
            trajectory_data[key] = np.array(trajectory_data[key])
        return trajectory_data

    def compute_global_pca_basis(self, pipe, force_recompute=False):
        """
        Compute global PCA basis from 1000 seed trajectories.

        This function generates trajectories for 1000 different seeds, collects all
        XT data across all inference steps, and computes a global PCA basis.
        The PCA basis vectors are saved to disk for reuse.

        Args:
            pipe: Loaded diffusion pipeline
            force_recompute: If True, recompute even if saved basis exists

        Returns:
            global_pca: PCA model fitted on all trajectory data
            pca_basis_vectors: The principal component vectors (basis vectors)
        """
        # Check if saved PCA basis exists
        pca_save_path = os.path.join(results_dir, f'global_pca_basis_{self.global_pca_postfix}.pkl')

        if os.path.exists(pca_save_path) and not force_recompute:
            print(f"\n📊 Loading saved global PCA basis from {pca_save_path}...")
            with open(pca_save_path, 'rb') as f:
                saved_data = pickle.load(f)
                global_pca = saved_data['pca']
                pca_basis_vectors = saved_data['basis_vectors']
                print(f"✓ Loaded global PCA basis (computed from {saved_data['num_seeds']} seeds)")
                return global_pca, pca_basis_vectors

        print(f"\n🚀 Computing global PCA basis from {self.num_global_seeds} seed trajectories...")
        print(f"   This may take a while...")

        # Collect all XT data from all seeds and all steps
        all_xt_data = []

        for i in range(self.num_global_seeds):
            if (i + 1) % 100 == 0:
                print(f"  Generating trajectory {i+1}/{self.num_global_seeds}")

            trajectory_data = self._generate_single_trajectory(pipe, i, self.model)
            # Collect all steps from this trajectory
            all_xt_data.append(trajectory_data['xt'])

        # Stack all data: shape will be (num_global_seeds * num_inference_steps, feature_dim)
        all_xt_data = np.vstack(all_xt_data)
        print(f"  Collected XT data shape: {all_xt_data.shape}")

        # Compute global PCA
        print(f"  Computing global PCA on all trajectory data...")
        global_pca = PCA(n_components=min(all_xt_data.shape[1], 100))  # Use up to 100 components
        global_pca.fit(all_xt_data)

        # Extract basis vectors (principal components)
        # components_ shape: (n_components, n_features)
        pca_basis_vectors = global_pca.components_

        print(f"  Global PCA - PC1: {global_pca.explained_variance_ratio_[0]:.4f}, "
              f"PC2: {global_pca.explained_variance_ratio_[1]:.4f}")
        print(f"  Total components: {len(global_pca.explained_variance_ratio_)}")

        # Save PCA basis for future use
        save_data = {
            'pca': global_pca,
            'basis_vectors': pca_basis_vectors,
            'num_seeds': self.num_global_seeds,
            'num_inference_steps': self.num_inference_steps,
            'explained_variance_ratio': global_pca.explained_variance_ratio_
        }
        with open(pca_save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"✓ Global PCA basis saved to: {pca_save_path}")

        return global_pca, pca_basis_vectors

    def project_single_seed_trajectory(self, pipe, global_pca):
        """
        Project a single seed trajectory onto the global PCA basis.

        For each inference step, project the XT vector onto the global PCA basis
        and extract the weights (coefficients) for the top 2 principal components.

        Args:
            pipe: Loaded diffusion pipeline
            global_pca: Global PCA model fitted on 1000 seed trajectories

        Returns:
            trajectory_data: Dictionary containing 'xt', 'timesteps'
            pc_weights: Array of shape (num_inference_steps, n_components) with PCA weights
            pc2_pc1_ratios: Array of PC2/PC1 weight ratios for each step
        """
        print(f"\n🚀 Generating single seed trajectory (seed={self.seed})...")

        # Generate single trajectory
        trajectory_data = self._generate_single_trajectory(pipe, self.seed, self.model)

        print(f"  Projecting trajectory onto global PCA basis...")

        # Project each step onto global PCA basis
        pc_weights = []  # Will store weights for each step
        pc2_pc1_ratios = []  # Will store PC2/PC1 ratio for each step

        for step in range(self.num_inference_steps):
            xt_step = trajectory_data['xt'][step]  # Shape: (feature_dim,)

            # Project onto global PCA basis
            # transform expects (n_samples, n_features), so we reshape
            xt_step_reshaped = xt_step.reshape(1, -1)
            projected = global_pca.transform(xt_step_reshaped)  # Shape: (1, n_components)

            # Get the weights (coefficients) for this step
            weights = projected[0]  # Shape: (n_components,)
            pc_weights.append(weights)

            # Calculate PC2/PC1 ratio using absolute values of weights
            # We use the top 2 components (PC1 and PC2)
            if len(weights) >= 2:
                pc1_weight = abs(weights[0])
                pc2_weight = abs(weights[1])
                ratio = pc2_weight / pc1_weight if pc1_weight > 0 else 0
                pc2_pc1_ratios.append(ratio)
            else:
                pc2_pc1_ratios.append(0)

        pc_weights = np.array(pc_weights)  # Shape: (num_inference_steps, n_components)
        pc2_pc1_ratios = np.array(pc2_pc1_ratios)  # Shape: (num_inference_steps,)

        print(f"✓ Projected trajectory onto global PCA basis")
        print(f"  PC weights shape: {pc_weights.shape}")
        print(f"  Mean PC2/PC1 ratio: {np.mean(pc2_pc1_ratios):.6f}")

        return trajectory_data, pc_weights, pc2_pc1_ratios

    def plot_xt_space_pca2(self, trajectory_data, global_pca, results_dir=results_dir):
        """Plot XT Space PCA2 Analysis using global PCA basis"""
        print(f"\n📊 Plotting XT Space PCA2...")

        # Project trajectory onto global PCA basis (using top 2 components)
        xt_pca_proj = global_pca.transform(trajectory_data['xt'])  # Shape: (num_steps, n_components)

        # Extract PC1 and PC2 (first two components)
        pc1_values = xt_pca_proj[:, 0]
        pc2_values = xt_pca_proj[:, 1]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Define colors
        start_color = '#27AE60'  # Green
        end_color = '#E67E22'    # Orange

        n_points = len(pc1_values)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))

        # Plot trajectory
        for j in range(n_points - 1):
            ax.plot([pc1_values[j], pc1_values[j+1]],
                   [pc2_values[j], pc2_values[j+1]],
                   color=colors[j], linewidth=3, alpha=0.9)

        # Mark start and end points
        ax.scatter(pc1_values[0], pc2_values[0],
                  c=start_color, s=200, marker='o', label='Start', zorder=5,
                  edgecolors='black', linewidth=2)
        ax.scatter(pc1_values[-1], pc2_values[-1],
                  c=end_color, s=200, marker='s', label='End', zorder=5,
                  edgecolors='black', linewidth=2)

        # Add step markers
        for j in range(0, n_points, max(1, n_points//8)):
            ax.scatter(pc1_values[j], pc2_values[j],
                      c=colors[j], s=80, marker='o', alpha=0.8, zorder=3)

        ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax.set_title('XT Space PCA2 Analysis', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()

        save_path = os.path.join(results_dir, f'xt_space_pca2_{self.pic_postfix_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ XT Space PCA2 plot saved to: {save_path}")

        plt.close()

    def plot_pc2_pc1_ratio(self, pc2_pc1_ratios, results_dir=results_dir):
        """Plot PC2/PC1 Ratio per Step"""
        print(f"\n📊 Plotting PC2/PC1 Ratio...")

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        method_color = '#3498DB'  # Blue

        steps = np.arange(self.num_inference_steps)

        # Sample points if too many
        if len(steps) > 40:
            sample_indices = np.linspace(0, len(steps)-1, 40, dtype=int)
            sampled_steps = steps[sample_indices]
            sampled_ratios = pc2_pc1_ratios[sample_indices]
        else:
            sampled_steps = steps
            sampled_ratios = pc2_pc1_ratios

        ax.plot(sampled_steps, sampled_ratios, 'o-', color=method_color, linewidth=3, markersize=6)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Theoretical (1.0)')
        ax.set_xlabel('Step', fontsize=12, fontweight='bold')

        # Set x-axis ticks
        if len(steps) > 100:
            max_step = steps[-1] + 2
            if max_step >= 1000:
                tick_interval = 200
            elif max_step >= 500:
                tick_interval = 100
            else:
                tick_interval = 50
            tick_steps = np.arange(0, max_step + 1, tick_interval)
            ax.set_xticks(tick_steps)
            ax.set_xticklabels([str(int(x)) for x in tick_steps])
        elif len(steps) > 10:
            max_step = steps[-1]
            tick_interval = max(10, max_step // 10)
            tick_steps = np.arange(0, max_step + 1, tick_interval)
            if tick_steps[-1] < max_step:
                tick_steps = np.append(tick_steps, max_step)
            ax.set_xticks(tick_steps)
            ax.set_xticklabels([str(int(x)) for x in tick_steps])
        else:
            ax.set_xticks(steps)
            ax.set_xticklabels([str(int(x)) for x in steps])

        ax.set_ylabel('PC2/PC1 Ratio', fontsize=12, fontweight='bold')
        ax.set_title('PC2/PC1 Ratio per Step', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = os.path.join(results_dir, f'pc2_pc1_ratio_{self.pic_postfix_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ PC2/PC1 Ratio plot saved to: {save_path}")

        plt.close()

def main(args):
    """
    Main function to run PCA2 analysis.

    This function performs the complete PCA2 analysis pipeline:
    1. Initialize analyzer with specified parameters
    2. Load the diffusion pipeline
    3. Compute global PCA basis from 1000 seed trajectories (or load if exists)
    4. Generate single seed trajectory and project onto global PCA basis
    5. Generate visualization plots (XT Space PCA2, PC2/PC1 Ratio)

    Args:
        args: Argument object containing:
            - method: Sampling method
            - model: Model type
            - num_inference_steps: Number of inference steps
            - num_global_seeds: Number of seeds for global PCA (default 1000)
            - seed: Random seed for single trajectory
            - force_recompute: Force recompute global PCA basis
    """
    analyzer = PCA2Analysis(
        method=args.method,
        model=args.model,
        num_inference_steps=args.num_inference_steps,
        num_global_seeds=args.num_global_seeds,
        seed=args.seed
    )

    try:
        # Step 1: Load the diffusion pipeline
        print(f"\n{'='*60}")
        print(f"Loading {args.method.upper()} {args.model.upper()} Pipeline")
        print(f"{'='*60}")
        pipe = analyzer.load_pipeline()

        # Step 2: Compute global PCA basis from 1000 seed trajectories
        print(f"\n{'='*60}")
        print(f"Computing Global PCA Basis from {args.num_global_seeds} Seeds")
        print(f"{'='*60}")
        global_pca, pca_basis_vectors = analyzer.compute_global_pca_basis(pipe, force_recompute=args.force_recompute)

        # Step 3: Generate single seed trajectory and project onto global PCA basis
        print(f"\n{'='*60}")
        print(f"Projecting Single Seed Trajectory (seed={args.seed})")
        print(f"{'='*60}")
        trajectory_data, pc_weights, pc2_pc1_ratios = analyzer.project_single_seed_trajectory(pipe, global_pca)

        # Step 4: Generate visualization plots
        print(f"\n{'='*60}")
        print(f"Creating PCA2 Analysis Plots")
        print(f"{'='*60}")
        analyzer.plot_xt_space_pca2(trajectory_data, global_pca)
        analyzer.plot_pc2_pc1_ratio(pc2_pc1_ratios)

        print(f"\n✅ PCA2 analysis completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 PCA2 Analysis (Optimized Version)")

    parser = argparse.ArgumentParser(description="PCA2 Analysis: Analyze diffusion trajectories using global PCA basis")
    parser.add_argument("--method", type=str, default="ddim",
                        help="Sampling method: ddim, dpm, dpm_lm, unipc, etc.")
    parser.add_argument("--model", type=str, default="ddpm_ema_cifar10",
                        choices=["ddpm_ema_cifar10", "ldm_celebahq_256", "stable-diffusion-2-base",
                                "stable-diffusion-xl-base-1.0", "stable-diffusion-v1-5"],
                        help="Model type to analyze")
    parser.add_argument("--num_inference_steps", type=int, default=100,
                        help="Number of inference steps in the diffusion process")
    parser.add_argument("--num_global_seeds", type=int, default=1000,
                        help="Number of seeds to use for computing global PCA basis")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for single trajectory analysis")
    parser.add_argument("--force_recompute", action='store_true',
                        help="Force recompute global PCA basis even if saved version exists")
    args = parser.parse_args()

    main(args)
