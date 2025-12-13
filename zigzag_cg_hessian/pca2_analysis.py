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
import glob

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

# Model color and marker mapping for visualization
MODEL_STYLES = {
    'ddpm_ema_cifar10': {
        'color': '#3498DB',  # Blue
        'marker': 'o',
        'linestyle': '-',
        'label': 'DDPM CIFAR-10',
        'linewidth': 3
    },
    'ldm_celebahq_256': {
        'color': '#E74C3C',  # Red
        'marker': 's',
        'linestyle': '-',
        'label': 'LDM CelebA-HQ',
        'linewidth': 3
    },
    'stable-diffusion-2-base': {
        'color': '#27AE60',  # Green
        'marker': '^',
        'linestyle': '-',
        'label': 'Stable Diffusion 2',
        'linewidth': 3
    },
    'stable-diffusion-xl-base-1.0': {
        'color': '#9B59B6',  # Purple
        'marker': 'D',
        'linestyle': '-',
        'label': 'Stable Diffusion XL',
        'linewidth': 3
    },
    'stable-diffusion-v1-5': {
        'color': '#F39C12',  # Orange
        'marker': 'v',
        'linestyle': '-',
        'label': 'Stable Diffusion v1.5',
        'linewidth': 3
    }
}

# Method color mapping
METHOD_COLORS = {
    'ddim': '#3498DB',      # Blue
    'dpm': '#E74C3C',       # Red
    'dpm_lm': '#27AE60',    # Green
    'dpm++': '#9B59B6',     # Purple
    'unipc': '#F39C12',     # Orange
    'pndm': '#1ABC9C',      # Turquoise
    'ddim_lm': '#E67E22'    # Dark Orange
}

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
                 num_global_seeds=1000, seeds=42, max_pca_components=None):
        """
        Initialize PCA2 Analysis.

        Args:
            method: Sampling method (ddim, dpm, dpm_lm, unipc, etc.)
            model: Model type (ddpm_ema_cifar10, ldm_celebahq_256, stable-diffusion-2-base, etc.)
            num_inference_steps: Number of inference steps in the diffusion process
            num_global_seeds: Number of seeds to use for computing global PCA basis
            seeds: List of random seeds for trajectory analysis (can be single seed or multiple seeds)
            max_pca_components: Maximum number of PCA components to compute.
                               If None, automatically determined based on feature dimension.
                               If feature_dim <= 500, uses all components.
                               If feature_dim > 500, uses min(feature_dim, 500).
        """
        self.method = method
        self.model = model
        self.num_inference_steps = num_inference_steps
        self.num_global_seeds = num_global_seeds
        # Ensure seeds is a list
        if isinstance(seeds, (list, tuple, np.ndarray)):
            self.seeds = list(seeds)
        else:
            self.seeds = [seeds]  # Single seed as list
        self.max_pca_components = max_pca_components

        # Generate postfix for output file names
        if len(self.seeds) == 1:
            self.pic_postfix_name = f'{self.model}_{self.method}_steps-{self.num_inference_steps}_seed-{self.seeds[0]}'
        else:
            # For multiple seeds, use range notation if consecutive, otherwise list first and last
            if len(self.seeds) <= 5:
                seeds_str = '_'.join(map(str, self.seeds))
            else:
                seeds_str = f'{self.seeds[0]}-{self.seeds[-1]}-{len(self.seeds)}seeds'
            self.pic_postfix_name = f'{self.model}_{self.method}_steps-{self.num_inference_steps}_seeds-{seeds_str}'
        self.global_pca_postfix = f'{self.model}_{self.method}_steps-{self.num_inference_steps}_global-{self.num_global_seeds}'

        print(f"🔧 PCA2 Analysis Configuration:")
        print(f"   - Method: {self.method}")
        print(f"   - Model: {self.model}")
        print(f"   - Inference steps per trajectory: {self.num_inference_steps}")
        print(f"   - Number of global seeds for PCA basis: {self.num_global_seeds}")
        print(f"   - Trajectory seeds: {self.seeds} ({len(self.seeds)} seed(s))")

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
        basis_vectors_save_path = os.path.join(results_dir, f'global_pca_basis_vectors_{self.global_pca_postfix}.npy')

        if os.path.exists(pca_save_path) and not force_recompute:
            print(f"\n📊 Checking for saved global PCA basis...")
            print(f"   File path: {pca_save_path}")

            try:
                with open(pca_save_path, 'rb') as f:
                    saved_data = pickle.load(f)

                # Verify that saved data matches current configuration
                saved_num_seeds = saved_data.get('num_seeds', None)
                saved_num_steps = saved_data.get('num_inference_steps', None)

                if saved_num_seeds == self.num_global_seeds and saved_num_steps == self.num_inference_steps:
                    global_pca = saved_data['pca']
                    pca_basis_vectors = saved_data['basis_vectors']

                    print(f"✓ Found matching saved global PCA basis!")
                    print(f"   - Computed from {saved_num_seeds} seeds")
                    print(f"   - Inference steps: {saved_num_steps}")
                    print(f"   - Basis vectors shape: {pca_basis_vectors.shape}")
                    print(f"   - PC1 explained variance: {global_pca.explained_variance_ratio_[0]:.4f}")
                    print(f"   - PC2 explained variance: {global_pca.explained_variance_ratio_[1]:.4f}")
                    print(f"   - Total components: {len(global_pca.explained_variance_ratio_)}")

                    # Also save basis vectors as separate numpy file for easy access
                    if not os.path.exists(basis_vectors_save_path):
                        np.save(basis_vectors_save_path, pca_basis_vectors)
                        print(f"✓ Saved basis vectors to: {basis_vectors_save_path}")

                    return global_pca, pca_basis_vectors
                else:
                    print(f"⚠️  Saved PCA basis configuration mismatch!")
                    print(f"   Saved: {saved_num_seeds} seeds, {saved_num_steps} steps")
                    print(f"   Current: {self.num_global_seeds} seeds, {self.num_inference_steps} steps")
                    print(f"   Will recompute with current configuration...")
            except Exception as e:
                print(f"⚠️  Error loading saved PCA basis: {e}")
                print(f"   Will recompute...")

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

        feature_dim = all_xt_data.shape[1]

        # Determine number of components to compute
        # For PCA basis representation, we want to capture the full dimensionality
        # However, for very high-dimensional data, we may need to limit for computational efficiency
        if self.max_pca_components is not None:
            # User specified maximum components
            n_components = min(feature_dim, self.max_pca_components)
            print(f"  Using user-specified max_pca_components: {self.max_pca_components}")
            print(f"  Feature dimension: {feature_dim}, will compute {n_components} components")
        else:
            # Automatic strategy:
            # 1. If feature_dim <= 500: use all components (complete basis)
            # 2. If feature_dim > 500: use min(feature_dim, 500) to balance completeness and efficiency
            #    (500 is chosen as a reasonable upper limit for most cases)
            # Note: We need enough components to properly represent the trajectory in PCA space
            if feature_dim <= 500:
                n_components = feature_dim  # Use all components for complete basis
                print(f"  Feature dimension ({feature_dim}) <= 500, using all components for complete basis")
            else:
                # For very high-dimensional data, limit to 500 components
                # This is a trade-off: we lose some information but gain computational efficiency
                # In practice, the first 500 components usually capture most of the variance
                n_components = 500
                print(f"  Feature dimension ({feature_dim}) > 500, using {n_components} components")
                print(f"  Note: This may lose some information, but first {n_components} components typically")
                print(f"        capture most of the variance in high-dimensional data")
                print(f"  Tip: Set max_pca_components={feature_dim} to use all components (slower but complete)")

        # Compute global PCA
        print(f"  Computing global PCA on all trajectory data...")
        print(f"  Number of components: {n_components}")
        global_pca = PCA(n_components=n_components)
        global_pca.fit(all_xt_data)

        # Extract basis vectors (principal components)
        # components_ shape: (n_components, n_features)
        pca_basis_vectors = global_pca.components_

        print(f"  Global PCA - PC1: {global_pca.explained_variance_ratio_[0]:.4f}, "
              f"PC2: {global_pca.explained_variance_ratio_[1]:.4f}")
        print(f"  Total components: {len(global_pca.explained_variance_ratio_)}")
        print(f"  Basis vectors shape: {pca_basis_vectors.shape}")

        # Save PCA basis for future use
        save_data = {
            'pca': global_pca,
            'basis_vectors': pca_basis_vectors,
            'num_seeds': self.num_global_seeds,
            'num_inference_steps': self.num_inference_steps,
            'n_components': n_components,
            'max_pca_components': self.max_pca_components,
            'feature_dim': feature_dim,
            'explained_variance_ratio': global_pca.explained_variance_ratio_,
            'method': self.method,
            'model': self.model
        }
        with open(pca_save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"✓ Global PCA basis saved to: {pca_save_path}")

        # Also save basis vectors as separate numpy file for easy access
        np.save(basis_vectors_save_path, pca_basis_vectors)
        print(f"✓ Basis vectors saved to: {basis_vectors_save_path}")

        return global_pca, pca_basis_vectors

    def project_single_seed_trajectory(self, pipe, global_pca):
        """
        Project a single seed trajectory onto the global PCA basis.

        For each inference step, project the XT vector onto the global PCA basis
        and find the two components with largest absolute weights.
        PC1 = component with largest absolute weight
        PC2 = component with second largest absolute weight

        Args:
            pipe: Loaded diffusion pipeline
            global_pca: Global PCA model fitted on 1000 seed trajectories

        Returns:
            trajectory_data: Dictionary containing 'xt', 'timesteps'
            pc_weights: Array of shape (num_inference_steps, n_components) with PCA weights
            pc2_pc1_ratios: Array of PC2/PC1 weight ratios for each step
            pc1_values_array: Array of PC1 values (largest weight for each step)
            pc2_values_array: Array of PC2 values (second largest weight for each step)
        """
        # Use first seed (should only be called when len(self.seeds) == 1)
        seed = self.seeds[0] if len(self.seeds) > 0 else 42
        print(f"\n🚀 Generating single seed trajectory (seed={seed})...")

        # Generate single trajectory
        trajectory_data = self._generate_single_trajectory(pipe, seed, self.model)

        print(f"  Projecting trajectory onto global PCA basis...")

        # Project each step onto global PCA basis
        pc_weights = []  # Will store weights for each step
        pc2_pc1_ratios = []  # Will store PC2/PC1 ratio for each step
        # Fixed: Use the first two principal components from global PCA (indices 0 and 1)
        # PC1 = first principal component (index 0)
        # PC2 = second principal component (index 1)
        pc1_indices = []  # Store which component is PC1 for each step (always 0)
        pc2_indices = []  # Store which component is PC2 for each step (always 1)
        pc1_values_list = []  # Store PC1 values (first PC weight)
        pc2_values_list = []  # Store PC2 values (second PC weight)

        for step in range(self.num_inference_steps):
            xt_step = trajectory_data['xt'][step]  # Shape: (feature_dim,)

            # Project onto global PCA basis
            # transform expects (n_samples, n_features), so we reshape
            xt_step_reshaped = xt_step.reshape(1, -1)
            projected = global_pca.transform(xt_step_reshaped)  # Shape: (1, n_components)

            # Get the weights (coefficients) for this step
            weights = projected[0]  # Shape: (n_components,)
            pc_weights.append(weights)

            # Dynamic selection: Find the two components with largest absolute weights
            # PC1 = component with largest absolute weight
            # PC2 = component with second largest absolute weight
            # Ensure |PC1| >= |PC2|
            if len(weights) >= 2:
                # Calculate absolute weights
                abs_weights = np.abs(weights)

                # Get indices of top 2 components by absolute weight (descending order)
                top2_indices = np.argsort(abs_weights)[-2:][::-1]
                pc1_idx = top2_indices[0]  # Largest absolute weight
                pc2_idx = top2_indices[1]  # Second largest absolute weight

                pc1_indices.append(pc1_idx)
                pc2_indices.append(pc2_idx)

                # Get the actual weight values
                pc1_weight_value = weights[pc1_idx]
                pc2_weight_value = weights[pc2_idx]

                # Use absolute values for PC1 and PC2
                pc1_abs = abs(pc1_weight_value)
                pc2_abs = abs(pc2_weight_value)

                # Ensure |PC1| >= |PC2| (should always be true, but double-check)
                if pc1_abs < pc2_abs:
                    # Swap if needed (shouldn't happen, but safety check)
                    print(f"⚠️  Warning at step {step}: |PC1|={pc1_abs:.6f} < |PC2|={pc2_abs:.6f}, swapping...")
                    pc1_idx, pc2_idx = pc2_idx, pc1_idx
                    pc1_abs, pc2_abs = pc2_abs, pc1_abs

                pc1_values_list.append(pc1_abs)  # Use absolute value
                pc2_values_list.append(pc2_abs)   # Use absolute value

                # Calculate PC2/PC1 ratio using absolute values
                ratio = pc2_abs / pc1_abs if pc1_abs > 0 else 0
                pc2_pc1_ratios.append(ratio)
            else:
                # Fallback if less than 2 components available
                if len(weights) >= 1:
                    pc1_abs = abs(weights[0])
                    pc2_abs = abs(weights[1]) if len(weights) > 1 else 0
                else:
                    pc1_abs = 0
                    pc2_abs = 0
                pc1_indices.append(0)
                pc2_indices.append(0 if len(weights) == 0 else 1)
                pc1_values_list.append(pc1_abs)
                pc2_values_list.append(pc2_abs)
                ratio = pc2_abs / pc1_abs if pc1_abs > 0 else 0
                pc2_pc1_ratios.append(ratio)

        pc_weights = np.array(pc_weights)  # Shape: (num_inference_steps, n_components)
        pc2_pc1_ratios = np.array(pc2_pc1_ratios)  # Shape: (num_inference_steps,)
        pc1_indices = np.array(pc1_indices)  # Shape: (num_inference_steps,)
        pc2_indices = np.array(pc2_indices)  # Shape: (num_inference_steps,)
        pc1_values_array = np.array(pc1_values_list)  # Shape: (num_inference_steps,)
        pc2_values_array = np.array(pc2_values_list)  # Shape: (num_inference_steps,)

        print(f"✓ Projected trajectory onto global PCA basis")
        print(f"  PC weights shape: {pc_weights.shape}")
        print(f"  Using dynamic PC1 and PC2 (largest and second largest absolute weights per step)")
        print(f"  PC1 indices range: [{pc1_indices.min()}, {pc1_indices.max()}]")
        print(f"  PC2 indices range: [{pc2_indices.min()}, {pc2_indices.max()}]")
        print(f"  Mean PC2/PC1 ratio: {np.mean(pc2_pc1_ratios):.6f}")

        return trajectory_data, pc_weights, pc2_pc1_ratios, pc1_values_array, pc2_values_array

    def project_multiple_seeds_trajectory(self, pipe, global_pca):
        """
        Project multiple seed trajectories onto the global PCA basis and compute statistics.

        For each seed, project the trajectory onto the global PCA basis using fixed
        PC1 (index 0) and PC2 (index 1). Then compute mean and std across all seeds.

        Args:
            pipe: Loaded diffusion pipeline
            global_pca: Global PCA model fitted on multiple seed trajectories

        Returns:
            all_trajectory_data: List of trajectory data dictionaries for each seed
            all_pc_weights: List of PC weights arrays for each seed
            pc2_pc1_ratios_mean: Mean PC2/PC1 ratios across seeds (shape: num_inference_steps,)
            pc2_pc1_ratios_std: Std PC2/PC1 ratios across seeds (shape: num_inference_steps,)
            pc1_values_mean: Mean PC1 values across seeds (shape: num_inference_steps,)
            pc1_values_std: Std PC1 values across seeds (shape: num_inference_steps,)
            pc2_values_mean: Mean PC2 values across seeds (shape: num_inference_steps,)
            pc2_values_std: Std PC2 values across seeds (shape: num_inference_steps,)
        """
        print(f"\n🚀 Generating multiple seed trajectories ({len(self.seeds)} seeds)...")

        all_trajectory_data = []
        all_pc_weights = []
        all_pc2_pc1_ratios = []
        all_pc1_values = []
        all_pc2_values = []

        for i, seed in enumerate(self.seeds):
            print(f"  Processing seed {seed} ({i+1}/{len(self.seeds)})...")

            # Generate trajectory for this seed
            trajectory_data = self._generate_single_trajectory(pipe, seed, self.model)
            all_trajectory_data.append(trajectory_data)

            # Project each step onto global PCA basis
            pc_weights = []
            pc2_pc1_ratios = []
            pc1_values_list = []
            pc2_values_list = []

            for step in range(self.num_inference_steps):
                xt_step = trajectory_data['xt'][step]  # Shape: (feature_dim,)

                # Project onto global PCA basis
                xt_step_reshaped = xt_step.reshape(1, -1)
                projected = global_pca.transform(xt_step_reshaped)  # Shape: (1, n_components)

                # Get the weights (coefficients) for this step
                weights = projected[0]  # Shape: (n_components,)
                pc_weights.append(weights)

                # Dynamic selection: Find the two components with largest absolute weights
                # PC1 = component with largest absolute weight
                # PC2 = component with second largest absolute weight
                # Ensure |PC1| >= |PC2|
                if len(weights) >= 2:
                    # Calculate absolute weights
                    abs_weights = np.abs(weights)

                    # Get indices of top 2 components by absolute weight (descending order)
                    top2_indices = np.argsort(abs_weights)[-2:][::-1]
                    pc1_idx = top2_indices[0]  # Largest absolute weight
                    pc2_idx = top2_indices[1]  # Second largest absolute weight

                    # Get the actual weight values
                    pc1_weight_value = weights[pc1_idx]
                    pc2_weight_value = weights[pc2_idx]

                    # Use absolute values for PC1 and PC2
                    pc1_abs = abs(pc1_weight_value)
                    pc2_abs = abs(pc2_weight_value)

                    # Ensure |PC1| >= |PC2| (should always be true, but double-check)
                    if pc1_abs < pc2_abs:
                        # Swap if needed (shouldn't happen, but safety check)
                        pc1_idx, pc2_idx = pc2_idx, pc1_idx
                        pc1_abs, pc2_abs = pc2_abs, pc1_abs

                    pc1_values_list.append(pc1_abs)
                    pc2_values_list.append(pc2_abs)

                    # Calculate PC2/PC1 ratio using absolute values
                    ratio = pc2_abs / pc1_abs if pc1_abs > 0 else 0
                    pc2_pc1_ratios.append(ratio)
                else:
                    # Fallback if less than 2 components available
                    if len(weights) >= 1:
                        pc1_abs = abs(weights[0])
                        pc2_abs = abs(weights[1]) if len(weights) > 1 else 0
                    else:
                        pc1_abs = 0
                        pc2_abs = 0
                    pc1_values_list.append(pc1_abs)
                    pc2_values_list.append(pc2_abs)
                    ratio = pc2_abs / pc1_abs if pc1_abs > 0 else 0
                    pc2_pc1_ratios.append(ratio)

            all_pc_weights.append(np.array(pc_weights))
            all_pc2_pc1_ratios.append(np.array(pc2_pc1_ratios))
            all_pc1_values.append(np.array(pc1_values_list))
            all_pc2_values.append(np.array(pc2_values_list))

        # Compute statistics across all seeds
        all_pc2_pc1_ratios = np.array(all_pc2_pc1_ratios)  # Shape: (num_seeds, num_inference_steps)
        all_pc1_values = np.array(all_pc1_values)  # Shape: (num_seeds, num_inference_steps)
        all_pc2_values = np.array(all_pc2_values)  # Shape: (num_seeds, num_inference_steps)

        pc2_pc1_ratios_mean = np.mean(all_pc2_pc1_ratios, axis=0)
        pc2_pc1_ratios_std = np.std(all_pc2_pc1_ratios, axis=0)
        pc1_values_mean = np.mean(all_pc1_values, axis=0)
        pc1_values_std = np.std(all_pc1_values, axis=0)
        pc2_values_mean = np.mean(all_pc2_values, axis=0)
        pc2_values_std = np.std(all_pc2_values, axis=0)

        print(f"✓ Projected {len(self.seeds)} trajectories onto global PCA basis")
        print(f"  Using dynamic PC1 and PC2 (largest and second largest absolute weights per step)")
        print(f"  Mean PC2/PC1 ratio across seeds: {np.mean(pc2_pc1_ratios_mean):.6f} ± {np.mean(pc2_pc1_ratios_std):.6f}")

        return (all_trajectory_data, all_pc_weights, pc2_pc1_ratios_mean, pc2_pc1_ratios_std,
                pc1_values_mean, pc1_values_std, pc2_values_mean, pc2_values_std)

    def plot_xt_space_pca2(self, trajectory_data, pc1_values_array, pc2_values_array,
                           pc1_values_std=None, pc2_values_std=None, results_dir=results_dir):
        """
        Plot XT Space PCA2 Analysis using dynamic PC1 and PC2 (largest and second largest absolute weights per step).

        Args:
            trajectory_data: Can be single dict or list of dicts (for multiple seeds)
            pc1_values_array: PC1 values (largest absolute weight per step, mean if multiple seeds)
            pc2_values_array: PC2 values (second largest absolute weight per step, mean if multiple seeds)
            pc1_values_std: Optional std for PC1 (for error bands)
            pc2_values_std: Optional std for PC2 (for error bands)
        """
        print(f"\n📊 Plotting XT Space PCA2...")

        # Use dynamic PC1 and PC2 values (largest and second largest absolute weights per step)
        pc1_values = pc1_values_array  # Largest absolute weight for each step (mean if multiple seeds)
        pc2_values = pc2_values_array   # Second largest absolute weight for each step (mean if multiple seeds)

        # Check if we have std for error bands
        has_error_bands = (pc1_values_std is not None) and (pc2_values_std is not None)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Get model style
        model_style = MODEL_STYLES.get(self.model, {
            'color': '#3498DB',
            'marker': 'o',
            'linestyle': '-',
            'label': self.model,
            'linewidth': 3
        })

        # Define colors
        start_color = '#27AE60'  # Green
        end_color = '#E67E22'    # Orange

        n_points = len(pc1_values)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))

        # Plot error bands if available (for multiple seeds)
        if has_error_bands:
            # Plot error bands as shaded regions
            ax.fill_between(pc1_values,
                           pc2_values - pc2_values_std,
                           pc2_values + pc2_values_std,
                           color=model_style['color'],
                           alpha=0.2,
                           zorder=0,
                           label=None)
            # Also show error in PC1 direction (approximate as ellipse)
            for j in range(0, n_points, max(1, n_points//10)):
                from matplotlib.patches import Ellipse
                ellipse = Ellipse((pc1_values[j], pc2_values[j]),
                                 width=2*pc1_values_std[j],
                                 height=2*pc2_values_std[j],
                                 color=model_style['color'],
                                 alpha=0.15,
                                 zorder=1)
                ax.add_patch(ellipse)

        # Plot trajectory with model-specific color (no label)
        trajectory_line = ax.plot(pc1_values, pc2_values,
                                  color=model_style['color'],
                                  linewidth=model_style['linewidth'],
                                  linestyle=model_style['linestyle'],
                                  alpha=0.7, zorder=2)

        # Plot trajectory segments with gradient colors
        for j in range(n_points - 1):
            ax.plot([pc1_values[j], pc1_values[j+1]],
                   [pc2_values[j], pc2_values[j+1]],
                   color=colors[j], linewidth=2, alpha=0.5, zorder=1)

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
        title = f'XT Space PCA2 Analysis - {model_style["label"]}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Auto-detect legend position to avoid overlap with data
        # Calculate data range and center
        x_range = pc1_values.max() - pc1_values.min()
        y_range = pc2_values.max() - pc2_values.min()
        x_center = (pc1_values.max() + pc1_values.min()) / 2
        y_center = (pc2_values.max() + pc2_values.min()) / 2

        # Define corner regions (30% of range from center toward each corner)
        # Upper-right: x > x_center + x_range*0.3, y > y_center + y_range*0.3
        # Upper-left: x < x_center - x_range*0.3, y > y_center + y_range*0.3
        ur_threshold_x = x_center + x_range * 0.3
        ur_threshold_y = y_center + y_range * 0.3
        ul_threshold_x = x_center - x_range * 0.3
        ul_threshold_y = y_center + y_range * 0.3

        # Count points in upper-right and upper-left corner regions
        ur_count = np.sum((pc1_values > ur_threshold_x) & (pc2_values > ur_threshold_y))
        ul_count = np.sum((pc1_values < ul_threshold_x) & (pc2_values > ul_threshold_y))

        # Choose legend position based on data density
        # If upper-right has more points, use upper-left, and vice versa
        if ur_count > ul_count:
            legend_loc = 'upper left'
        else:
            legend_loc = 'upper right'

        # Legend with only Start and End, in one row
        ax.legend(fontsize=10, loc=legend_loc, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()

        save_path = os.path.join(results_dir, f'xt_space_pca2_{self.pic_postfix_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ XT Space PCA2 plot saved to: {save_path}")

        plt.close()

    def plot_pc2_pc1_ratio(self, pc2_pc1_ratios, pc2_pc1_ratios_std=None, results_dir=results_dir):
        """
        Plot PC2/PC1 Ratio per Step

        Args:
            pc2_pc1_ratios: Mean PC2/PC1 ratios (if multiple seeds) or single ratios
            pc2_pc1_ratios_std: Optional std for error bands
        """
        print(f"\n📊 Plotting PC2/PC1 Ratio...")

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Get model and method styles
        model_style = MODEL_STYLES.get(self.model, {
            'color': '#3498DB',
            'marker': 'o',
            'linestyle': '-',
            'label': self.model,
            'linewidth': 3
        })
        method_color = METHOD_COLORS.get(self.method, '#3498DB')

        steps = np.arange(self.num_inference_steps)

        # Sample points if too many
        if len(steps) > 40:
            sample_indices = np.linspace(0, len(steps)-1, 40, dtype=int)
            sampled_steps = steps[sample_indices]
            sampled_ratios = pc2_pc1_ratios[sample_indices]
            if pc2_pc1_ratios_std is not None:
                sampled_ratios_std = pc2_pc1_ratios_std[sample_indices]
            else:
                sampled_ratios_std = None
        else:
            sampled_steps = steps
            sampled_ratios = pc2_pc1_ratios
            sampled_ratios_std = pc2_pc1_ratios_std

        # Plot error bands if available (for multiple seeds)
        if sampled_ratios_std is not None:
            ax.fill_between(sampled_steps,
                           sampled_ratios - sampled_ratios_std,
                           sampled_ratios + sampled_ratios_std,
                           color=model_style['color'],
                           alpha=0.2,
                           zorder=0)

        # Use model color for the main line (no label)
        ax.plot(sampled_steps, sampled_ratios,
               marker=model_style['marker'],
               linestyle=model_style['linestyle'],
               color=model_style['color'],
               linewidth=model_style['linewidth'],
               markersize=6)
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
        title = f'PC2/PC1 Ratio per Step - {model_style["label"]}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = os.path.join(results_dir, f'pc2_pc1_ratio_{self.pic_postfix_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ PC2/PC1 Ratio plot saved to: {save_path}")

        plt.close()

    def generate_legend(self, results_dir=results_dir):
        """
        Generate and save a legend figure for the current model.

        Args:
            results_dir: Directory to save the legend figure
        """
        print(f"\n📊 Generating legend for {self.model}...")

        # Get model style
        model_style = MODEL_STYLES.get(self.model, {
            'color': '#3498DB',
            'marker': 'o',
            'linestyle': '-',
            'label': self.model,
            'linewidth': 3
        })

        # Create figure for legend only
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off')

        # Create legend entries
        legend_elements = [
            plt.Line2D([0], [0],
                      color=model_style['color'],
                      marker=model_style['marker'],
                      linestyle=model_style['linestyle'],
                      linewidth=model_style['linewidth'],
                      markersize=10,
                      label=model_style['label']),
            plt.Line2D([0], [0],
                      color='#27AE60',
                      marker='o',
                      linestyle='None',
                      markersize=12,
                      label='Start Point'),
            plt.Line2D([0], [0],
                      color='#E67E22',
                      marker='s',
                      linestyle='None',
                      markersize=12,
                      label='End Point'),
            plt.Line2D([0], [0],
                      color='red',
                      linestyle='--',
                      linewidth=2,
                      label='Theoretical (1.0)')
        ]

        # Add method information if available
        if self.method in METHOD_COLORS:
            method_color = METHOD_COLORS[self.method]
            legend_elements.append(
                plt.Line2D([0], [0],
                          color=method_color,
                          linestyle='-',
                          linewidth=2,
                          label=f'Method: {self.method.upper()}')
            )

        # Create legend - all items in one row
        num_items = len(legend_elements)
        legend = ax.legend(handles=legend_elements,
                         loc='center',
                         fontsize=12,
                         frameon=True,
                         fancybox=True,
                         shadow=True,
                         ncol=num_items)  # All items in one row

        # Set title
        title = f'Legend - {model_style["label"]}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.95)

        plt.tight_layout()

        # Save legend
        legend_path = os.path.join(results_dir, f'legend_{self.pic_postfix_name}.png')
        plt.savefig(legend_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ Legend saved to: {legend_path}")

        plt.close()

    def save_analysis_data(self, trajectory_data, pc_weights, pc2_pc1_ratios, pc1_values_array, pc2_values_array, global_pca,
                           pc2_pc1_ratios_std=None, pc1_values_std=None, pc2_values_std=None, results_dir=results_dir):
        """
        Save analysis data for later use in generating combined plots.

        Args:
            trajectory_data: Dictionary containing 'xt', 'timesteps' (or list of dicts for multiple seeds)
            pc_weights: Array of PCA weights (or list of arrays for multiple seeds)
            pc2_pc1_ratios: Array of PC2/PC1 ratios (mean if multiple seeds)
            pc1_values_array: Array of PC1 values (mean if multiple seeds)
            pc2_values_array: Array of PC2 values (mean if multiple seeds)
            global_pca: Global PCA model
            pc2_pc1_ratios_std: Optional std for PC2/PC1 ratios (for multiple seeds)
            pc1_values_std: Optional std for PC1 values (for multiple seeds)
            pc2_values_std: Optional std for PC2 values (for multiple seeds)
            results_dir: Directory to save the data
        """
        data_save_path = os.path.join(results_dir, f'analysis_data_{self.pic_postfix_name}.pkl')

        save_data = {
            'model': self.model,
            'method': self.method,
            'num_inference_steps': self.num_inference_steps,
            'seeds': self.seeds,
            'num_seeds': len(self.seeds),
            'trajectory_data': trajectory_data,
            'pc_weights': pc_weights,
            'pc2_pc1_ratios': pc2_pc1_ratios,
            'pc1_values_array': pc1_values_array,
            'pc2_values_array': pc2_values_array,
            'global_pca': global_pca
        }

        # Add std if available (multiple seeds)
        if pc2_pc1_ratios_std is not None:
            save_data['pc2_pc1_ratios_std'] = pc2_pc1_ratios_std
        if pc1_values_std is not None:
            save_data['pc1_values_std'] = pc1_values_std
        if pc2_values_std is not None:
            save_data['pc2_values_std'] = pc2_values_std

        with open(data_save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"✓ Analysis data saved to: {data_save_path}")

def parse_seeds(seed_str):
    """
    Parse seed string into a list of integers.

    Supports:
    - Single number: "42"
    - Comma-separated: "42,43,44"
    - Range: "42-50" (inclusive)
    - Mixed: "42,45-50,60"
    """
    seeds = []
    parts = seed_str.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range format: start-end
            start, end = part.split('-')
            start = int(start.strip())
            end = int(end.strip())
            seeds.extend(range(start, end + 1))
        else:
            # Single number
            seeds.append(int(part))

    return sorted(list(set(seeds)))  # Remove duplicates and sort

def main(args):
    """
    Main function to run PCA2 analysis.

    This function performs the complete PCA2 analysis pipeline:
    1. Initialize analyzer with specified parameters
    2. Load the diffusion pipeline
    3. Compute global PCA basis from 1000 seed trajectories (or load if exists)
    4. Generate trajectory(ies) and project onto global PCA basis
    5. Generate visualization plots (XT Space PCA2, PC2/PC1 Ratio)

    Args:
        args: Argument object containing:
            - method: Sampling method
            - model: Model type
            - num_inference_steps: Number of inference steps
            - num_global_seeds: Number of seeds for global PCA (default 1000)
            - seed: Random seed(s) for trajectory analysis (can be single or multiple)
            - force_recompute: Force recompute global PCA basis
    """
    # Parse seeds
    seeds = parse_seeds(args.seed)

    analyzer = PCA2Analysis(
        method=args.method,
        model=args.model,
        num_inference_steps=args.num_inference_steps,
        num_global_seeds=args.num_global_seeds,
        seeds=seeds,
        max_pca_components=args.max_pca_components
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

        # Step 3: Generate trajectory(ies) and project onto global PCA basis
        if len(seeds) == 1:
            # Single seed: use original function
            print(f"\n{'='*60}")
            print(f"Projecting Single Seed Trajectory (seed={seeds[0]})")
            print(f"{'='*60}")
            trajectory_data, pc_weights, pc2_pc1_ratios, pc1_values_array, pc2_values_array = analyzer.project_single_seed_trajectory(pipe, global_pca)

            # Step 4: Generate visualization plots
            print(f"\n{'='*60}")
            print(f"Creating PCA2 Analysis Plots")
            print(f"{'='*60}")
            analyzer.plot_xt_space_pca2(trajectory_data, pc1_values_array, pc2_values_array)
            analyzer.plot_pc2_pc1_ratio(pc2_pc1_ratios)

            # Step 5: Save analysis data
            print(f"\n{'='*60}")
            print(f"Saving Analysis Data")
            print(f"{'='*60}")
            analyzer.save_analysis_data(trajectory_data, pc_weights, pc2_pc1_ratios, pc1_values_array, pc2_values_array, global_pca)
        else:
            # Multiple seeds: use new function with statistics
            print(f"\n{'='*60}")
            print(f"Projecting Multiple Seed Trajectories (seeds={seeds})")
            print(f"{'='*60}")
            (all_trajectory_data, all_pc_weights, pc2_pc1_ratios_mean, pc2_pc1_ratios_std,
             pc1_values_mean, pc1_values_std, pc2_values_mean, pc2_values_std) = analyzer.project_multiple_seeds_trajectory(pipe, global_pca)

            # Step 4: Generate visualization plots with error bands
            print(f"\n{'='*60}")
            print(f"Creating PCA2 Analysis Plots (with error bands)")
            print(f"{'='*60}")
            # Use first trajectory for reference (or mean trajectory)
            analyzer.plot_xt_space_pca2(all_trajectory_data[0], pc1_values_mean, pc2_values_mean,
                                       pc1_values_std=pc1_values_std, pc2_values_std=pc2_values_std)
            analyzer.plot_pc2_pc1_ratio(pc2_pc1_ratios_mean, pc2_pc1_ratios_std=pc2_pc1_ratios_std)

            # Step 5: Save analysis data
            print(f"\n{'='*60}")
            print(f"Saving Analysis Data")
            print(f"{'='*60}")
            # For multiple seeds, save mean and std
            analyzer.save_analysis_data(all_trajectory_data, all_pc_weights,
                                       pc2_pc1_ratios_mean, pc1_values_mean, pc2_values_mean,
                                       global_pca, pc2_pc1_ratios_std=pc2_pc1_ratios_std,
                                       pc1_values_std=pc1_values_std, pc2_values_std=pc2_values_std)

        # Step 6: Generate legend for current model
        print(f"\n{'='*60}")
        print(f"Generating Legend")
        print(f"{'='*60}")
        analyzer.generate_legend()

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
    parser.add_argument("--model", type=str, default="ldm_celebahq_256",
                        choices=["ddpm_ema_cifar10", "ldm_celebahq_256", "stable-diffusion-2-base",
                                "stable-diffusion-xl-base-1.0", "stable-diffusion-v1-5"],
                        help="Model type to analyze")
    parser.add_argument("--num_inference_steps", type=int, default=200,
                        help="Number of inference steps in the diffusion process")
    parser.add_argument("--num_global_seeds", type=int, default=200,
                        help="Number of seeds to use for computing global PCA basis")
    parser.add_argument("--seed", type=str, default="67-101",
                        help="Random seed(s) for trajectory analysis. "
                             "Can be: single number (e.g., 42), "
                             "comma-separated list (e.g., 42,43,44), "
                             "or range (e.g., 42-50)")
    parser.add_argument("--max_pca_components", type=int, default=None,
                        help="Maximum number of PCA components to compute. "
                             "If None, automatically determined: uses all components if feature_dim <= 500, "
                             "otherwise uses 500. For complete basis representation, set to feature_dim.")
    parser.add_argument("--force_recompute", action='store_true',
                        help="Force recompute global PCA basis even if saved version exists")
    args = parser.parse_args()

    main(args)
