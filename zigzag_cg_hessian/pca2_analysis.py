#!/usr/bin/env python3
"""
PCA2 Analysis: XT Space PCA2, PC2/PC1 Ratio, Score Space PCA2
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import sys
import os
import argparse
from datetime import datetime

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
    in both the XT space (image state space) and Score space (gradient space).
    It generates visualizations and statistics to analyze the anisotropy and
    evolution of diffusion trajectories.

    Key analyses:
    - XT Space PCA2: 2D projection of image state evolution
    - Score Space PCA2: 2D projection of gradient/score evolution
    - PC2/PC1 Ratio: Measures anisotropy at each inference step
    - Local trajectory windows: Detailed view at high-slope points
    """

    def __init__(self, method='ddim', model='ddpm_ema_cifar10', num_inference_steps=1000, num_trajectories=20, seed=42):
        """
        Initialize PCA2 Analysis.

        Args:
            method: Sampling method (ddim, dpm, dpm_lm, unipc, etc.)
            model: Model type (ddpm_ema_cifar10, ldm_celebahq_256, stable-diffusion-2-base, etc.)
            num_inference_steps: Number of inference steps in the diffusion process
            num_trajectories: Number of trajectories to generate for statistical analysis
            seed: Random seed for reproducibility
        """
        self.method = method
        self.model = model
        self.num_inference_steps = num_inference_steps
        self.num_trajectories = num_trajectories
        self.seed = seed

        # Generate postfix for output file names
        # Format: {model}_{method}_steps-{num_inference_steps}_trajs-{num_trajectories}_seed-{seed}
        self.pic_postfix_name = f'{self.model}_{self.method}_steps-{self.num_inference_steps}_trajs-{self.num_trajectories}_seed-{self.seed}'

        print(f"🔧 PCA2 Analysis Configuration:")
        print(f"   - Method: {self.method}")
        print(f"   - Model: {self.model}")
        print(f"   - Inference steps per trajectory: {self.num_inference_steps}")
        print(f"   - Number of trajectories: {self.num_trajectories}")
        print(f"   - Seed: {self.seed}")

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
            pipe.vqvae.config.scaling_factor = 1.0 # args.scaling_factor
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
        # 获取原始配置并过滤掉运行时状态属性（避免警告）
        original_config = pipe.scheduler.config
        config_dict = {
            k: v for k, v in original_config.items()
            if k not in ['timestep_values', 'timesteps']  # 移除这些运行时状态属性
        }

        if sampler_type == 'pndm':
            pipe.scheduler = PNDMScheduler.from_config(config_dict)
            project.info(f"  Using PNDM scheduler")

        elif sampler_type == 'ddim':
            pipe.scheduler = DDIMScheduler.from_config(config_dict)
            pipe.scheduler.config.eta = 0.0  # 设置eta=0.0，与celeba_test.py保持一致
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
            # Fallback prompts for testing
            return {
                "00001": "a beautiful landscape with mountains and trees",
                "00002": "a cat sitting on a windowsill",
                "00003": "a modern city skyline at sunset",
                "00004": "a vintage car parked on the street",
                "00005": "a delicious plate of pasta"
            }

    def generate_trajectories(self, pipe):
        """
        Generate multiple trajectories for statistical analysis.

        Each trajectory contains intermediate states at each inference step:
        - xt: Image state vector at step t
        - score: Score/drift vector (negative of noise prediction)
        - timesteps: Timestep values
        - noise_pred: Noise prediction from the UNet

        Args:
            pipe: Loaded diffusion pipeline

        Returns:
            trajectories: List of trajectory dictionaries, each containing the above fields
        """
        print(f"\n🚀 Generating {self.num_trajectories} trajectories using {self.method.upper()}...")

        trajectories = []
        # Set base seed for reproducibility
        torch.manual_seed(self.seed)

        # Generate multiple trajectories with different seeds (seed, seed+1, seed+2, ...)
        for i in range(self.num_trajectories):
            if i % 10 == 0:
                print(f"  Generating trajectory {i+1}/{self.num_trajectories}")

            # Each trajectory uses a different seed (seed + i) for variation
            trajectory_data = self._generate_single_trajectory(pipe, self.seed + i, self.model)
            trajectories.append(trajectory_data)

        print(f"✓ Generated {len(trajectories)} trajectories")
        return trajectories

    def _generate_single_trajectory(self, pipe, seed, model_type="ddpm_ema_cifar10"):
        """
        Generate a single trajectory based on model type.

        Args:
            pipe: Pipeline instance
            seed: Random seed for trajectory generation
            model_type: Type of model ('ddpm_ema_cifar10', 'ldm_celebahq_256',
                      'stable-diffusion-2-base', etc.)

        Returns:
            trajectory_data: Dictionary containing 'xt', 'score', 'timesteps', 'noise_pred'
        """
        torch.manual_seed(seed)

        # Load COCO prompts for Stable Diffusion models
        # For text-to-image models, we need prompts for generation
        coco_prompts_dict = self.load_coco_prompts()
        if seed in coco_prompts_dict:
            prompt = coco_prompts_dict[seed]
        else:
            # Default prompt if seed not found in COCO prompts
            prompt = "a beautiful landscape with mountains and trees"

        # Route to appropriate trajectory generation function based on model type
        if model_type == 'ddpm_ema_cifar10':
            # CIFAR-10 unconditional generation (no prompt needed)
            trajectory_data = self._generate_single_trajectory_cifar10(pipe, seed)
        elif model_type == 'ldm_celebahq_256':
            # CelebA-HQ unconditional generation (no prompt needed)
            trajectory_data = self._generate_single_trajectory_celeba(pipe, seed)
        elif model_type in ['stable-diffusion-2-base', 'stable-diffusion-xl-base-1.0', 'stable-diffusion-v1-5']:
            # Stable Diffusion text-to-image generation (requires prompt)
            trajectory_data = self._generate_single_trajectory_sd(pipe, seed, prompt)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return trajectory_data

    def _generate_single_trajectory_cifar10(self, pipe, seed):
        """Generate a single trajectory and collect intermediate states"""
        torch.manual_seed(seed)

        # Initialize random noise
        device = pipe.unet.device
        shape = (1, pipe.unet.config.in_channels, 32, 32)
        latents = torch.randn(shape, device=device)

        # Store trajectory data
        trajectory_data = {
            'xt': [],      # 待生成的图片状态向量
            'score': [],   # 分数/漂移向量
            'timesteps': [],
            'noise_pred': []
        }

        scheduler = pipe.scheduler

        for j, t in enumerate(scheduler.timesteps):
            # Store current state
            xt_flat = latents.detach().view(1, -1).cpu().numpy().flatten()
            trajectory_data['xt'].append(xt_flat)
            trajectory_data['timesteps'].append(t.item())

            # Predict noise
            with torch.no_grad():
                noise_pred = pipe.unet(latents, t).sample

            # Store noise prediction
            noise_pred_flat = noise_pred.detach().view(1, -1).cpu().numpy().flatten()
            trajectory_data['noise_pred'].append(noise_pred_flat)

            # Compute score/drift vector
            score = -noise_pred_flat
            trajectory_data['score'].append(score)

            # DDIM sampling step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Convert to numpy arrays
        for key in ['xt', 'score', 'timesteps', 'noise_pred']:
            trajectory_data[key] = np.array(trajectory_data[key])

        return trajectory_data

    def _generate_single_trajectory_celeba(self, pipe, seed):
        """
        Generate a single trajectory for CelebA pipeline and optionally collect intermediate states.
        Compatible with _generate_single_trajectory from ddim_iclr_1x3_analysis.py

        Args:
            pipe: LDMPipeline instance
            seed: Random seed
        Returns:
            trajectory_data
        """
        torch.manual_seed(seed)
        device = pipe.unet.device

        # Initialize latents (for LDM, we need to get the latent shape from VAE)
        # LDM uses VAE encoder, but for generation we start with random latents
        height = pipe.vqvae.config.sample_size if hasattr(pipe.vqvae.config, 'sample_size') else 256
        width = height
        latent_channels = pipe.unet.config.in_channels
        shape = (1, latent_channels, height // 8, width // 8)  # VAE downsampling factor is 8
        latents = torch.randn(shape, device=device, dtype=pipe.unet.dtype)

        # Set timesteps
        pipe.scheduler.set_timesteps(self.num_inference_steps, device=device)

        # Store trajectory data
        trajectory_data = {
            'xt': [],      # 待生成的图片状态向量
            'score': [],   # 分数/漂移向量
            'timesteps': [],
            'noise_pred': []
        }

        scheduler = pipe.scheduler

        # Manual sampling loop (similar to _generate_single_trajectory)
        for j, t in enumerate(scheduler.timesteps):
            # Store current state
            xt_flat = latents.detach().view(1, -1).cpu().numpy().flatten()
            trajectory_data['xt'].append(xt_flat)
            trajectory_data['timesteps'].append(t.item() if isinstance(t, torch.Tensor) else t)

            # Predict noise
            with torch.no_grad():
                noise_pred = pipe.unet(latents, t).sample

            # Store noise prediction
            noise_pred_flat = noise_pred.detach().view(1, -1).cpu().numpy().flatten()
            trajectory_data['noise_pred'].append(noise_pred_flat)

            # Compute score/drift vector
            score = -noise_pred_flat
            trajectory_data['score'].append(score)

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Convert trajectory data to numpy arrays
        for key in ['xt', 'score', 'timesteps', 'noise_pred']:
            trajectory_data[key] = np.array(trajectory_data[key])
        return trajectory_data

    def _generate_single_trajectory_sd(self, pipe, seed,
                                    prompt="a beautiful landscape with mountains and trees", negative_prompt="", guidance_scale=7.5):
        """
        Generate a single trajectory for Stable Diffusion pipeline and optionally collect intermediate states.
        Compatible with _generate_single_trajectory from ddim_iclr_1x3_analysis.py

        Args:
            pipe: StableDiffusionPipeline or StableDiffusionXLPipeline instance
            prompt: Text prompt
            negative_prompt: Negative text prompt
            seed: Random seed
            guidance_scale: Guidance scale for classifier-free guidance

        Returns:
            trajectory_data
        """
        torch.manual_seed(seed)
        device = pipe.device

        # Prepare text embeddings
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

        # Prepare negative prompt embeddings if provided
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

        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Initialize latents
        # Get VAE scale factor (8 for SD, may vary for SDXL)
        vae_scale_factor = getattr(pipe, 'vae_scale_factor', 8)
        if hasattr(pipe.unet.config, 'sample_size'):
            height = width = pipe.unet.config.sample_size * vae_scale_factor
        else:
            # Default to 512 for standard SD, 1024 for SDXL
            height = width = 512 if 'xl' not in pipe.__class__.__name__.lower() else 1024
        latent_channels = pipe.unet.config.in_channels
        shape = (1, latent_channels, height // vae_scale_factor, width // vae_scale_factor)
        latents = torch.randn(shape, device=device, dtype=text_embeddings.dtype)

        # Set timesteps
        pipe.scheduler.set_timesteps(self.num_inference_steps, device=device)

        # Scale latents
        latents = latents * pipe.scheduler.init_noise_sigma

        # Store trajectory data
        trajectory_data = {
            'xt': [],      # 待生成的图片状态向量
            'score': [],   # 分数/漂移向量
            'timesteps': [],
            'noise_pred': []
        }

        scheduler = pipe.scheduler

        # Manual sampling loop (similar to _generate_single_trajectory)
        for j, t in enumerate(scheduler.timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # Store current state
            xt_flat = latents.detach().view(1, -1).cpu().numpy().flatten()
            trajectory_data['xt'].append(xt_flat)
            trajectory_data['timesteps'].append(t.item() if isinstance(t, torch.Tensor) else t)

            # Predict noise
            with torch.no_grad():
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample

            # Perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Store the final guided noise prediction
            noise_pred_flat = noise_pred.detach().view(1, -1).cpu().numpy().flatten()
            trajectory_data['noise_pred'].append(noise_pred_flat)

            # Compute score/drift vector
            score = -noise_pred_flat
            trajectory_data['score'].append(score)

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # # Decode latents to images using VAE
        # from PIL import Image
        # with torch.no_grad():
        #     latents = 1 / pipe.vae.config.scaling_factor * latents
        #     image = pipe.vae.decode(latents).sample
        #     image = (image / 2 + 0.5).clamp(0, 1)
        #     image = image.cpu().permute(0, 2, 3, 1).numpy()
        #     image = (image * 255).round().astype("uint8")
        #     image = Image.fromarray(image[0])

        # Convert trajectory data to numpy arrays
        for key in ['xt', 'score', 'timesteps', 'noise_pred']:
            trajectory_data[key] = np.array(trajectory_data[key])
        return trajectory_data

    def create_pca_models(self, trajectories):
        """Create PCA models for both XT and Score analysis"""
        print(f"\n📊 Creating PCA models...")

        # Collect XT data
        xt_data = []
        for traj in trajectories:
            xt_data.append(traj['xt'])
        xt_data = np.vstack(xt_data)
        print(f"  XT data shape: {xt_data.shape}")

        # Collect Score data
        score_data = []
        for traj in trajectories:
            score_data.append(traj['score'])
        score_data = np.vstack(score_data)
        print(f"  Score data shape: {score_data.shape}")

        # Create PCA models
        xt_pca = PCA(n_components=2, svd_solver='randomized')
        xt_pca.fit(xt_data)
        print(f"  XT PCA - PC1: {xt_pca.explained_variance_ratio_[0]:.4f}, PC2: {xt_pca.explained_variance_ratio_[1]:.4f}")

        score_pca = PCA(n_components=2, svd_solver='randomized')
        score_pca.fit(score_data)
        print(f"  Score PCA - PC1: {score_pca.explained_variance_ratio_[0]:.4f}, PC2: {score_pca.explained_variance_ratio_[1]:.4f}")

        return xt_pca, score_pca, xt_data, score_data

    def calculate_pc2_pc1_ratio_per_step(self, trajectories, xt_pca):
        """
        Calculate PC2/PC1 ratio for each inference step.

        This ratio measures the anisotropy of the trajectory at each step.
        - Ratio = 1.0: Isotropic (equal variance in PC1 and PC2)
        - Ratio < 1.0: More variance in PC1 (anisotropic)
        - Ratio > 1.0: More variance in PC2 (anisotropic)

        For each step, we perform PCA on all trajectories at that step,
        then compute the ratio of explained variances.

        Args:
            trajectories: List of trajectory dictionaries
            xt_pca: Global PCA model (not used here, but kept for consistency)

        Returns:
            xt_step_ratios: Array of PC2/PC1 ratios, one for each inference step
        """
        print(f"\n📊 Calculating PC2/PC1 ratio per step...")

        # Collect all step data
        xt_step_ratios = []
        for step in range(self.num_inference_steps):
            # Collect XT data from all trajectories at this step
            step_data = []
            for traj in trajectories:
                step_data.append(traj['xt'][step])

            step_data = np.array(step_data)

            # Calculate PCA for this specific step
            # This gives us the local anisotropy at this step
            step_pca = PCA(n_components=2, svd_solver='randomized')
            step_pca.fit(step_data)

            # Calculate PC2/PC1 ratio from explained variances
            # explained_variance_ is the variance explained by each component
            pc1_var = step_pca.explained_variance_[0]
            pc2_var = step_pca.explained_variance_[1]
            ratio = pc2_var / pc1_var if pc1_var > 0 else 0

            xt_step_ratios.append(ratio)

            # Print progress every 5 steps
            if step % 5 == 0:
                print(f"  Step {step}: PC2/PC1 = {ratio:.6f}")

        return np.array(xt_step_ratios)

    def find_high_slope_points(self, xt_step_ratios, num_points=3):
        """Find points with high slope changes in PC2/PC1 ratio, one from each third of the trajectory"""
        # Calculate slope (first derivative)
        slopes = np.diff(xt_step_ratios)

        # Divide trajectory into 3 equal parts
        total_steps = len(xt_step_ratios)
        part_size = total_steps // 3

        # Define three intervals: early, middle, late
        intervals = [
            (0, part_size),
            (part_size, 2 * part_size),
            (2 * part_size, total_steps)
        ]

        # Find the point with highest slope in each interval
        selected_points = []
        for i, (start, end) in enumerate(intervals):
            # Ensure we have enough space for 7-point window (3 points on each side)
            interval_start = max(start, 3)
            interval_end = min(end, total_steps - 4)

            if interval_start < interval_end:
                # Get slopes in this interval
                interval_slopes = slopes[interval_start:interval_end]
                if len(interval_slopes) > 0:
                    # Find the index with highest absolute slope in this interval
                    local_max_idx = np.argmax(np.abs(interval_slopes))
                    global_idx = interval_start + local_max_idx
                    selected_points.append(global_idx)
                    print(f"  Interval {i+1} [{interval_start}:{interval_end}]: selected t={global_idx}, slope={slopes[global_idx]:.6f}")

        return selected_points[:num_points]

    def plot_xt_space_pca2(self, trajectories, xt_pca, results_dir=results_dir):
        """Plot XT Space PCA2 Analysis as a separate figure"""

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Define colors
        start_color = '#27AE60'  # Green
        end_color = '#E67E22'    # Orange

        # XT Space PCA2 Analysis
        traj = trajectories[0]  # Use first trajectory
        xt_pca_proj = xt_pca.transform(traj['xt'])
        n_points = len(xt_pca_proj)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))

        # Plot trajectory
        for j in range(n_points - 1):
            ax.plot([xt_pca_proj[j, 0], xt_pca_proj[j+1, 0]],
                   [xt_pca_proj[j, 1], xt_pca_proj[j+1, 1]],
                   color=colors[j], linewidth=3, alpha=0.9)

        # Mark start and end points
        ax.scatter(xt_pca_proj[0, 0], xt_pca_proj[0, 1],
                  c=start_color, s=200, marker='o', label='Start', zorder=5,
                  edgecolors='black', linewidth=2)
        ax.scatter(xt_pca_proj[-1, 0], xt_pca_proj[-1, 1],
                  c=end_color, s=200, marker='s', label='End', zorder=5,
                  edgecolors='black', linewidth=2)

        # Add step markers
        for j in range(0, n_points, max(1, n_points//8)):
            ax.scatter(xt_pca_proj[j, 0], xt_pca_proj[j, 1],
                      c=colors[j], s=80, marker='o', alpha=0.8, zorder=3)

        ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax.set_title('XT Space PCA2 Analysis', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(results_dir, f'xt_space_pca2_{self.pic_postfix_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ XT Space PCA2 plot saved to: {save_path}")

        plt.close()

    def plot_pc2_pc1_ratio(self, xt_step_ratios, results_dir=results_dir):
        """Plot PC2/PC1 Ratio per Step as a separate figure"""
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Define colors
        method_color = '#3498DB'  # Blue

        # PC2/PC1 Ratio per Step (最多显示40个点)
        steps = np.arange(self.num_inference_steps)

        # 如果步数超过40，则均匀采样40个点
        if len(steps) > 40:
            sample_indices = np.linspace(0, len(steps)-1, 40, dtype=int)
            sampled_steps = steps[sample_indices]
            sampled_ratios = xt_step_ratios[sample_indices]
        else:
            sampled_steps = steps
            sampled_ratios = xt_step_ratios

        ax.plot(sampled_steps, sampled_ratios, 'o-', color=method_color, linewidth=3, markersize=6)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Theoretical (1.0)')
        ax.set_xlabel('Step', fontsize=12, fontweight='bold')

        # 设置x轴刻度 - 显示整十或整百的刻度
        if len(steps) > 100:
            # 对于大范围，使用整百刻度
            max_step = steps[-1] + 2
            if max_step >= 1000:
                tick_interval = 200
            elif max_step >= 500:
                tick_interval = 100
            else:
                tick_interval = 50

            # 生成整百/整十刻度
            tick_steps = np.arange(0, max_step + 1, tick_interval)

            ax.set_xticks(tick_steps)
            ax.set_xticklabels([str(int(x)) for x in tick_steps])
        elif len(steps) > 10:
            # 对于中等范围，使用整十刻度
            max_step = steps[-1]
            tick_interval = max(10, max_step // 10)
            tick_steps = np.arange(0, max_step + 1, tick_interval)
            if tick_steps[-1] < max_step:
                tick_steps = np.append(tick_steps, max_step)

            ax.set_xticks(tick_steps)
            ax.set_xticklabels([str(int(x)) for x in tick_steps])
        else:
            # 对于小范围，显示所有刻度
            ax.set_xticks(steps)
            ax.set_xticklabels([str(int(x)) for x in steps])

        ax.set_ylabel('PC2/PC1 Ratio', fontsize=12, fontweight='bold')
        ax.set_title('PC2/PC1 Ratio per Step', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(results_dir, f'pc2_pc1_ratio_{self.pic_postfix_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ PC2/PC1 Ratio plot saved to: {save_path}")

        plt.close()

    def plot_score_space_pca2(self, trajectories, score_pca, results_dir=results_dir):
        """Plot Score Space PCA2 Analysis as a separate figure"""
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Define colors
        start_color = '#27AE60'  # Green
        end_color = '#E67E22'    # Orange

        # Score Space PCA2 Analysis
        traj = trajectories[0]  # Use first trajectory
        score_pca_proj = score_pca.transform(traj['score'])
        n_points = len(score_pca_proj)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))

        # Plot trajectory
        for j in range(n_points - 1):
            ax.plot([score_pca_proj[j, 0], score_pca_proj[j+1, 0]],
                   [score_pca_proj[j, 1], score_pca_proj[j+1, 1]],
                   color=colors[j], linewidth=3, alpha=0.9)

        # Mark start and end points
        ax.scatter(score_pca_proj[0, 0], score_pca_proj[0, 1],
                  c=start_color, s=200, marker='o', label='Start', zorder=5,
                  edgecolors='black', linewidth=2)
        ax.scatter(score_pca_proj[-1, 0], score_pca_proj[-1, 1],
                  c=end_color, s=200, marker='s', label='End', zorder=5,
                  edgecolors='black', linewidth=2)

        # Add step markers
        for j in range(0, n_points, max(1, n_points//8)):
            ax.scatter(score_pca_proj[j, 0], score_pca_proj[j, 1],
                      c=colors[j], s=80, marker='o', alpha=0.8, zorder=3)

        # 自适应调整x轴显示范围，让图例占满坐标轴的至少2/3
        x_data = score_pca_proj[:, 0]
        y_data = score_pca_proj[:, 1]

        # 计算数据范围
        x_range = np.max(x_data) - np.min(x_data)
        y_range = np.max(y_data) - np.min(y_data)

        # 计算中心点
        x_center = (np.max(x_data) + np.min(x_data)) / 2
        y_center = (np.max(y_data) + np.min(y_data)) / 2

        # 计算显示范围，确保图例占满坐标轴的至少2/3
        display_ratio = 0.67  # 至少2/3
        x_display_range = x_range / display_ratio
        y_display_range = y_range / display_ratio

        # 设置坐标轴范围
        ax.set_xlim(x_center - x_display_range/2, x_center + x_display_range/2)
        ax.set_ylim(y_center - y_display_range/2, y_center + y_display_range/2)

        ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax.set_title('Score Space PCA2 Analysis', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(results_dir, f'score_space_pca2_{self.pic_postfix_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ Score Space PCA2 plot saved to: {save_path}")

        plt.close()

    def plot_iclr_1x3_analysis(self, trajectories, xt_pca, score_pca, xt_step_ratios, results_dir=results_dir):
        """Plot ICLR 1x3 analysis for PCA2 - now calls three separate plot functions"""

        print(f"\n📊 Generating separate PCA2 analysis plots...")

        # Plot and save each figure separately
        self.plot_xt_space_pca2(trajectories, xt_pca, results_dir)
        self.plot_pc2_pc1_ratio(xt_step_ratios, results_dir)
        self.plot_score_space_pca2(trajectories, score_pca, results_dir)

        print(f"\n✓ All PCA2 analysis plots saved separately")

        # 生成局部轨迹图
        self.plot_local_trajectory_windows(trajectories, xt_pca, xt_step_ratios, results_dir)

    def plot_local_trajectory_windows(self, trajectories, xt_pca, xt_step_ratios, results_dir=results_dir):
        """Generate local trajectory windows at high slope points"""
        print(f"\n📊 Generating local trajectory windows...")

        # 找到高斜率变化点
        high_slope_points = self.find_high_slope_points(xt_step_ratios, num_points=3)
        print(f"  High slope points: {high_slope_points}")

        # 使用第一条轨迹进行局部窗口分析
        traj = trajectories[0]
        xt_pca_proj = xt_pca.transform(traj['xt'])

        # 创建局部窗口图
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle('DDIM Local Trajectory Windows (7-point around high slope changes)',
                     fontsize=14, fontweight='bold', y=0.95)

        for i, t in enumerate(high_slope_points):
            ax = axes[i]
            self.plot_local_window(xt_pca_proj, t, ax, f"Local 7-step around t={t}")

        plt.tight_layout()

        # 保存局部窗口图
        local_windows_path = os.path.join(results_dir, f'local_windows_{self.pic_postfix_name}.png')
        plt.savefig(local_windows_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ {self.pic_postfix_name} local trajectory windows plot saved to: {local_windows_path}")

        plt.close()

    def plot_local_window(self, Z2d, t, ax, title):
        """
        在给定 Axes 上绘制 [t-3, t-2, t-1, t, t+1, t+2, t+3] 的投影折线
        Z2d: [T, 2] 的投影坐标
        """
        assert 3 <= t <= Z2d.shape[0]-4, f"t={t} 超出可取 7 帧窗口的范围"
        idx = np.arange(t-3, t+4)
        pts = Z2d[idx]  # [7, 2]

        ax.plot(pts[:, 0], pts[:, 1], "-o", linewidth=2, markersize=5)
        # 用方块标注中心帧 t
        ax.scatter(pts[3, 0], pts[3, 1], s=120, marker='s', zorder=3, label=f"t={t}")
        ax.legend(frameon=False, loc="best")
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.axis("equal")
        ax.grid(True, linestyle=":")

        # 设置 x 和 y 轴只显示 3 个刻度，保留4位有效数字
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_ticks = [x_min, (x_min + x_max) / 2, x_max]
        y_ticks = [y_min, (y_min + y_max) / 2, y_max]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x:.4g}" for x in x_ticks])
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{y:.4g}" for y in y_ticks])

    def generate_analysis_report(self, trajectories, xt_pca, score_pca, xt_step_ratios, results_dir=results_dir):
        """
        Generate analysis report with detailed statistics and arrays.

        Args:
            trajectories: List of trajectory dictionaries (needed to calculate per-step PC1/PC2 values)
            xt_pca: Global PCA model for XT space
            score_pca: Global PCA model for Score space
            xt_step_ratios: Array of PC2/PC1 ratios for each step
            results_dir: Directory to save the report
        """
        report_path = os.path.join(results_dir, f'pca2_analysis_report_{self.pic_postfix_name}.txt')

        # Calculate per-step PC1 and PC2 true values
        print(f"\n📊 Calculating per-step PC1/PC2 values for report...")
        xt_step_pc1_values = []
        xt_step_pc2_values = []
        for step in range(self.num_inference_steps):
            step_data = []
            for traj in trajectories:
                step_data.append(traj['xt'][step])
            step_data = np.array(step_data)
            step_pca = PCA(n_components=2, svd_solver='randomized')
            step_pca.fit(step_data)
            xt_step_pc1_values.append(step_pca.explained_variance_[0])
            xt_step_pc2_values.append(step_pca.explained_variance_[1])

        xt_step_pc1_values = np.array(xt_step_pc1_values)
        xt_step_pc2_values = np.array(xt_step_pc2_values)

        with open(report_path, 'w') as f:
            f.write(f"{self.method} {self.model} PCA2 Analysis Report\n")
            f.write("="*50 + "\n\n")

            f.write("EXPERIMENTAL CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Method: {self.method.upper()}\n")
            f.write(f"Trajectories: {self.num_trajectories}\n")
            f.write(f"Inference steps per trajectory: {self.num_inference_steps}\n")
            f.write(f"Total data points: {self.num_trajectories * self.num_inference_steps}\n\n")

            f.write("XT SPACE PCA2 ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"PC1 explained variance ratio: {xt_pca.explained_variance_ratio_[0]:.6f}\n")
            f.write(f"PC2 explained variance ratio: {xt_pca.explained_variance_ratio_[1]:.6f}\n")
            f.write(f"PC1 explained variance (true value): {xt_pca.explained_variance_[0]:.6f}\n")
            f.write(f"PC2 explained variance (true value): {xt_pca.explained_variance_[1]:.6f}\n")
            f.write(f"Total explained variance: {np.sum(xt_pca.explained_variance_ratio_):.6f}\n")
            f.write(f"PC1/PC2 ratio: {xt_pca.explained_variance_[0]/xt_pca.explained_variance_[1]:.6f}\n\n")

            f.write("SCORE SPACE PCA2 ANALYSIS:\n")
            f.write("-" * 28 + "\n")
            f.write(f"PC1 explained variance ratio: {score_pca.explained_variance_ratio_[0]:.6f}\n")
            f.write(f"PC2 explained variance ratio: {score_pca.explained_variance_ratio_[1]:.6f}\n")
            f.write(f"PC1 explained variance (true value): {score_pca.explained_variance_[0]:.6f}\n")
            f.write(f"PC2 explained variance (true value): {score_pca.explained_variance_[1]:.6f}\n")
            f.write(f"Total explained variance: {np.sum(score_pca.explained_variance_ratio_):.6f}\n")
            f.write(f"PC1/PC2 ratio: {score_pca.explained_variance_[0]/score_pca.explained_variance_[1]:.6f}\n\n")

            f.write("PC2/PC1 RATIO PER STEP ANALYSIS:\n")
            f.write("-" * 32 + "\n")
            f.write(f"Mean ratio: {np.mean(xt_step_ratios):.6f}\n")
            f.write(f"Std ratio: {np.std(xt_step_ratios):.6f}\n")
            f.write(f"Min ratio: {np.min(xt_step_ratios):.6f}\n")
            f.write(f"Max ratio: {np.max(xt_step_ratios):.6f}\n")
            f.write(f"Deviation from 1.0: {np.mean(np.abs(xt_step_ratios - 1.0)):.6f}\n\n")

            # 添加高斜率点信息
            high_slope_points = self.find_high_slope_points(xt_step_ratios, num_points=3)
            f.write("HIGH SLOPE POINTS FOR LOCAL WINDOWS:\n")
            f.write("-" * 35 + "\n")
            for i, t in enumerate(high_slope_points):
                slope = np.diff(xt_step_ratios)[t-1] if t > 0 else 0
                f.write(f"Point {i+1}: t={t}, slope={slope:.6f}\n")
            f.write("\n")

            # Write arrays in a more readable format (one value per line)
            # This format is easy to parse programmatically
            f.write("ARRAYS (ONE VALUE PER LINE):\n")
            f.write("-" * 35 + "\n")

            # XT Step PC1 values array
            f.write("XT_STEP_PC1_VALUES:\n")
            f.write(f"# Array length: {len(xt_step_pc1_values)}\n")
            f.write(f"# Format: one value per line\n")
            for val in xt_step_pc1_values:
                f.write(f"{val:.6f}\n")
            f.write("\n")

            # XT Step PC2 values array
            f.write("XT_STEP_PC2_VALUES:\n")
            f.write(f"# Array length: {len(xt_step_pc2_values)}\n")
            f.write(f"# Format: one value per line\n")
            for val in xt_step_pc2_values:
                f.write(f"{val:.6f}\n")
            f.write("\n")

            # XT Step Ratios (PC2/PC1) array
            f.write("XT_STEP_RATIOS (PC2/PC1):\n")
            f.write(f"# Array length: {len(xt_step_ratios)}\n")
            f.write(f"# Format: one value per line\n")
            for ratio in xt_step_ratios:
                f.write(f"{ratio:.6f}\n")
            f.write("\n")

            # Also write step-by-step format for backward compatibility
            f.write("STEP-BY-STEP DETAILED VIEW:\n")
            f.write("-" * 30 + "\n")
            f.write("# Format: Step | PC1 | PC2 | PC2/PC1 Ratio\n")
            for i in range(self.num_inference_steps):
                f.write(f"Step {i:2d}: PC1={xt_step_pc1_values[i]:.6f}, PC2={xt_step_pc2_values[i]:.6f}, Ratio={xt_step_ratios[i]:.6f}\n")

        print(f"\n✓ Analysis report saved to: {report_path}")

        # Print summary to console
        print(f"\n" + "="*80)
        print("DDIM ICLR 1x3 ANALYSIS SUMMARY")
        print("="*80)

        print(f"\nXT Space PCA2:")
        print(f"PC1 explained variance ratio: {xt_pca.explained_variance_ratio_[0]:.6f}")
        print(f"PC2 explained variance ratio: {xt_pca.explained_variance_ratio_[1]:.6f}")
        print(f"PC1 explained variance (true value): {xt_pca.explained_variance_[0]:.6f}")
        print(f"PC2 explained variance (true value): {xt_pca.explained_variance_[1]:.6f}")
        print(f"PC1/PC2 ratio: {xt_pca.explained_variance_[0]/xt_pca.explained_variance_[1]:.6f}")

        print(f"\nScore Space PCA2:")
        print(f"PC1 explained variance ratio: {score_pca.explained_variance_ratio_[0]:.6f}")
        print(f"PC2 explained variance ratio: {score_pca.explained_variance_ratio_[1]:.6f}")
        print(f"PC1 explained variance (true value): {score_pca.explained_variance_[0]:.6f}")
        print(f"PC2 explained variance (true value): {score_pca.explained_variance_[1]:.6f}")
        print(f"PC1/PC2 ratio: {score_pca.explained_variance_[0]/score_pca.explained_variance_[1]:.6f}")

        print(f"\nPC2/PC1 Ratio per Step:")
        print(f"Mean: {np.mean(xt_step_ratios):.6f}")
        print(f"Std: {np.std(xt_step_ratios):.6f}")
        print(f"Deviation from 1.0: {np.mean(np.abs(xt_step_ratios - 1.0)):.6f}")

        # 打印高斜率点信息
        high_slope_points = self.find_high_slope_points(xt_step_ratios, num_points=3)
        print(f"\nHigh Slope Points for Local Windows:")
        for i, t in enumerate(high_slope_points):
            slope = np.diff(xt_step_ratios)[t-1] if t > 0 else 0
            print(f"  Point {i+1}: t={t}, slope={slope:.6f}")

def main(args):
    """
    Main function to run PCA2 analysis.

    This function performs the complete PCA2 analysis pipeline:
    1. Initialize analyzer with specified parameters
    2. Load the diffusion pipeline
    3. Generate multiple trajectories
    4. Create PCA models for XT and Score spaces
    5. Calculate PC2/PC1 ratio per step
    6. Generate visualization plots (separate figures for each analysis)
    7. Generate analysis report

    Args:
        args: Argument object containing:
            - method: Sampling method (ddim, dpm, dpm_lm, unipc, etc.)
            - model: Model type (ddpm_ema_cifar10, ldm_celebahq_256, stable-diffusion-2-base, etc.)
            - num_inference_steps: Number of inference steps
            - num_trajectories: Number of trajectories to generate
            - seed: Random seed for reproducibility
    """
    # Initialize analyzer with specified parameters
    analyzer = PCA2Analysis(
        method=args.method,
        model=args.model,
        num_inference_steps=args.num_inference_steps,
        num_trajectories=args.num_trajectories,
        seed=args.seed
    )

    try:
        # Step 1: Load the diffusion pipeline
        print(f"\n{'='*60}")
        print(f"Loading {args.method.upper()} {args.model.upper()} Pipeline")
        print(f"{'='*60}")
        pipe = analyzer.load_pipeline()

        # Step 2: Generate multiple trajectories for statistical analysis
        # Each trajectory contains intermediate states (xt, score, timesteps, noise_pred)
        print(f"\n{'='*60}")
        print(f"Generating {args.num_trajectories} Trajectories")
        print(f"{'='*60}")
        trajectories = analyzer.generate_trajectories(pipe)

        # Step 3: Create PCA models for both XT space and Score space
        # PCA is performed on all trajectory data to find principal components
        print(f"\n{'='*60}")
        print(f"Creating PCA Models for {args.method.upper()} {args.model.upper()}")
        print(f"{'='*60}")
        xt_pca, score_pca, xt_data, score_data = analyzer.create_pca_models(trajectories)

        # Step 4: Calculate PC2/PC1 ratio for each inference step
        # This ratio indicates the anisotropy of the trajectory at each step
        print(f"\n{'='*60}")
        print(f"Calculating PC2/PC1 Ratio per Step for {args.method.upper()} {args.model.upper()}")
        print(f"{'='*60}")
        xt_step_ratios = analyzer.calculate_pc2_pc1_ratio_per_step(trajectories, xt_pca)

        # Step 5: Generate visualization plots
        # Creates three separate figures: XT Space PCA2, PC2/PC1 Ratio, Score Space PCA2
        print(f"\n{'='*60}")
        print(f"Creating PCA2 Analysis Plots for {args.method.upper()} {args.model.upper()}")
        print(f"{'='*60}")
        analyzer.plot_iclr_1x3_analysis(trajectories, xt_pca, score_pca, xt_step_ratios)

        # Step 6: Generate detailed analysis report
        # Saves statistics and step-by-step ratios to a text file
        print(f"\n{'='*60}")
        print(f"Generating Analysis Report for {args.method.upper()} {args.model.upper()}")
        print(f"{'='*60}")
        analyzer.generate_analysis_report(trajectories, xt_pca, score_pca, xt_step_ratios)

        print(f"\n✅ PCA2 analysis completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    """
    Main entry point for PCA2 Analysis.

    This script can be run in two modes:
    1. Single run: Use command-line arguments to run a single analysis
    2. Batch run: Automatically runs multiple configurations (commented out by default)

    Usage examples:
        # Single run with default parameters
        python pca2_analysis.py

        # Single run with custom parameters
        python pca2_analysis.py --method ddim --model ddpm_ema_cifar10 --num_inference_steps 1000 --num_trajectories 20 --seed 42
    """
    print("🚀 PCA2 Analysis")

    # Default configuration (used if command-line args not provided)
    # These can be overridden by command-line arguments
    method_name = 'ddim'
    # model_type = 'ddpm_ema_cifar10'
    model_type = 'ldm_celebahq_256'
    # model_type = 'stable-diffusion-2-base'
    # model_type = 'stable-diffusion-xl-base-1.0'
    # model_type = 'stable-diffusion-v1-5'
    num_inference_steps = 1000
    num_trajectories = 10
    seed = 32

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="PCA2 Analysis: Analyze diffusion trajectories using PCA")
    parser.add_argument("--method", type=str, default="ddim",
                        help="Sampling method: ddim, dpm, dpm_lm, unipc, etc.")
    parser.add_argument("--model", type=str, default="ddpm_ema_cifar10",
                        choices=["ddpm_ema_cifar10", "ldm_celebahq_256", "stable-diffusion-2-base",
                                "stable-diffusion-xl-base-1.0", "stable-diffusion-v1-5"],
                        help="Model type to analyze")
    parser.add_argument("--num_inference_steps", type=int, default=200,
                        help="Number of inference steps in the diffusion process")
    parser.add_argument("--num_trajectories", type=int, default=20,
                        help="Number of trajectories to generate for statistical analysis")
    parser.add_argument("--seed", type=int, default=12,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Run single analysis with provided arguments
    main(args)

    # ============================================================================
    # BATCH RUN MODE (commented out by default)
    # Uncomment the following sections to run batch experiments
    # ============================================================================

    # Batch run 1: CIFAR-10 and CelebA-HQ models
    print("Batch run 1: CIFAR-10 and CelebA-HQ models")
    # These models support more inference steps (100-1000)
    for num_inference_steps in [1000, 200, 500, 100]:
       for method in ["ddim", "dpm", "dpm_lm", "unipc"]:
            for num_trajectories in [60, 10, 6, 20, 40]:
                for model in ["ddpm_ema_cifar10", "ldm_celebahq_256"]:
                    print(f"Running: method: {method}, model: {model}, num_inference_steps: {num_inference_steps}, num_trajectories: {num_trajectories}")
                    args.method = method
                    args.model = model
                    args.num_inference_steps = num_inference_steps
                    args.num_trajectories = num_trajectories
                    main(args)

    # Batch run 2: Stable Diffusion models
    print("Batch run 2: Stable Diffusion models")
    # These models typically use fewer inference steps (10-200)
    for num_inference_steps in [100, 20, 50, 10, 200]:
        for method in ["ddim", "dpm", "dpm_lm", "unipc"]:
            for num_trajectories in [60, 10, 6, 20, 40]:
                for model in ["stable-diffusion-2-base", "stable-diffusion-xl-base-1.0", "stable-diffusion-v1-5"]:
                    print(f"Running: method: {method}, model: {model}, num_inference_steps: {num_inference_steps}, num_trajectories: {num_trajectories}")
                    args.method = method
                    args.model = model
                    args.num_inference_steps = num_inference_steps
                    args.num_trajectories = num_trajectories
                    main(args)