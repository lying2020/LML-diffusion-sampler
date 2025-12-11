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
    """PCA2 Analysis"""

    def __init__(self, n_samples=10, num_inference_steps=1000, num_trajectories=20):
        self.n_samples = n_samples
        self.num_inference_steps = num_inference_steps
        self.num_trajectories = num_trajectories
        self.method = 'ddim'
        self.model = 'ddpm_ema_cifar10'

        print(f"🔧 PCA2 Analysis Configuration:")
        print(f"   - CIFAR-10 samples for PCA: {self.n_samples}")
        print(f"   - Inference steps per trajectory: {self.num_inference_steps}")
        print(f"   - Number of trajectories: {self.num_trajectories}")

    def load_pipeline(self, method_name='ddim', model_type='ddpm_ema_cifar10'):

        self.model = model_type
        self.method = method_name
        print(f"\n🔧 Loading {method_name.upper()} {self.model.upper()} pipeline...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_type == 'ddpm_ema_cifar10':
            pipe = DDPMPipeline.from_pretrained(cifar10_model_path, torch_dtype=torch.float32, use_safetensors=False)
            pipe.unet.to(device)
            self.setup_cifar10_scheduler(pipe, method_name)

        elif model_type == 'ldm_celebahq_256':
            pipe = LDMPipeline.from_pretrained(celeba_model_path, torch_dtype=torch.float32, use_safetensors=False)
            pipe.unet.to(device)
            pipe.vqvae.to(device)
            pipe.vqvae.config.scaling_factor = 1.0 # args.scaling_factor
            self.setup_celeba_scheduler(pipe, method_name)

        elif model_type == 'stable-diffusion-2-base':
            pipe = StableDiffusionPipeline.from_pretrained(coco_sd2_model_path, torch_dtype=torch.float32, use_safetensors=False)
            pipe.unet.to(device)
            self.setup_sd_scheduler(pipe, method_name)

        elif model_type == 'stable-diffusion-xl-base-1.0':
            pipe = StableDiffusionXLPipeline.from_pretrained(coco_sdxl_model_path, torch_dtype=torch.float32, use_safetensors=False, safety_checker=None, added_cond_kwargs={})
            pipe.unet.to(device)
            self.setup_sd_scheduler(pipe, method_name)

        elif model_type == 'stable-diffusion-v1-5':
            pipe = StableDiffusionPipeline.from_pretrained(coco_sd15_model_path, torch_dtype=torch.float32, use_safetensors=False)
            pipe.unet.to(device)
            self.setup_sd_scheduler(pipe, method_name)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

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


    def generate_trajectories(self, pipe, seed=42):
        """Generate multiple trajectories for statistical analysis"""
        print(f"\n🚀 Generating {self.num_trajectories} trajectories using {self.method.upper()}...")

        trajectories = []
        torch.manual_seed(seed)

        for i in range(self.num_trajectories):
            if i % 10 == 0:
                print(f"  Generating trajectory {i+1}/{self.num_trajectories}")

            trajectory_data = self._generate_single_trajectory(pipe, seed + i, self.model)
            trajectories.append(trajectory_data)

        print(f"✓ Generated {len(trajectories)} trajectories")
        return trajectories

    def _generate_single_trajectory(self, pipe, seed, model_type="ddpm_ema_cifar10"):
        if model_type == 'ddpm_ema_cifar10':
            trajectory_data = self._generate_single_trajectory_cifar10(pipe, seed)
        elif model_type == 'ldm_celebahq_256':
            trajectory_data = self._generate_single_trajectory_celeba(pipe, seed)
        elif model_type == 'stable-diffusion-2-base':
            trajectory_data = self._generate_single_trajectory_sd(pipe, seed)
        elif model_type == 'stable-diffusion-xl-base-1.0':
            trajectory_data = self._generate_single_trajectory_sd(pipe, seed)
        elif model_type == 'stable-diffusion-v1-5':
            trajectory_data = self._generate_single_trajectory_sd(pipe, seed)
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
        """Calculate PC2/PC1 ratio for each step"""
        print(f"\n📊 Calculating PC2/PC1 ratio per step...")

        # Collect all step data
        step_ratios = []
        for step in range(self.num_inference_steps):
            step_data = []
            for traj in trajectories:
                step_data.append(traj['xt'][step])

            step_data = np.array(step_data)

            # Calculate PCA for this step
            step_pca = PCA(n_components=2, svd_solver='randomized')
            step_pca.fit(step_data)

            # Calculate PC2/PC1 ratio
            pc1_var = step_pca.explained_variance_[0]
            pc2_var = step_pca.explained_variance_[1]
            ratio = pc2_var / pc1_var if pc1_var > 0 else 0

            step_ratios.append(ratio)

            if step % 5 == 0:
                print(f"  Step {step}: PC2/PC1 = {ratio:.6f}")

        return np.array(step_ratios)

    def find_high_slope_points(self, step_ratios, num_points=3):
        """Find points with high slope changes in PC2/PC1 ratio, one from each third of the trajectory"""
        # Calculate slope (first derivative)
        slopes = np.diff(step_ratios)

        # Divide trajectory into 3 equal parts
        total_steps = len(step_ratios)
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

    def plot_iclr_1x3_analysis(self, trajectories, xt_pca, score_pca, step_ratios, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """Plot ICLR 1x3 analysis for PCA2"""
        os.makedirs(save_dir, exist_ok=True)

        # Create figure with 1x3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle('PCA2 Analysis: XT Space PCA2, PC2/PC1 Ratio',
                     fontsize=16, fontweight='bold', y=0.95)

        # Define colors
        start_color = '#27AE60'  # Green
        end_color = '#E67E22'    # Orange
        method_color = '#3498DB'  # Blue for DDIM

        # Subplot 1: XT Space PCA2 Analysis
        ax1 = axes[0]
        traj = trajectories[0]  # Use first trajectory
        xt_pca_proj = xt_pca.transform(traj['xt'])
        n_points = len(xt_pca_proj)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))

        # Plot trajectory
        for j in range(n_points - 1):
            ax1.plot([xt_pca_proj[j, 0], xt_pca_proj[j+1, 0]],
                   [xt_pca_proj[j, 1], xt_pca_proj[j+1, 1]],
                   color=colors[j], linewidth=3, alpha=0.9)

        # Mark start and end points
        ax1.scatter(xt_pca_proj[0, 0], xt_pca_proj[0, 1],
                  c=start_color, s=200, marker='o', label='Start', zorder=5,
                  edgecolors='black', linewidth=2)
        ax1.scatter(xt_pca_proj[-1, 0], xt_pca_proj[-1, 1],
                  c=end_color, s=200, marker='s', label='End', zorder=5,
                  edgecolors='black', linewidth=2)

        # Add step markers
        for j in range(0, n_points, max(1, n_points//8)):
            ax1.scatter(xt_pca_proj[j, 0], xt_pca_proj[j, 1],
                      c=colors[j], s=80, marker='o', alpha=0.8, zorder=3)

        ax1.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax1.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax1.set_title('XT Space PCA2 Analysis', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # Subplot 2: PC2/PC1 Ratio per Step (最多显示40个点)
        ax2 = axes[1]
        steps = np.arange(self.num_inference_steps)

        # 如果步数超过40，则均匀采样40个点
        if len(steps) > 40:
            sample_indices = np.linspace(0, len(steps)-1, 40, dtype=int)
            sampled_steps = steps[sample_indices]
            sampled_ratios = step_ratios[sample_indices]
        else:
            sampled_steps = steps
            sampled_ratios = step_ratios

        ax2.plot(sampled_steps, sampled_ratios, 'o-', color=method_color, linewidth=3, markersize=6)
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Theoretical (1.0)')
        ax2.set_xlabel('Step', fontsize=12, fontweight='bold')

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
            # # 确保包含最后一个点
            # if tick_steps[-1] < max_step:
            #     tick_steps = np.append(tick_steps, max_step)

            ax2.set_xticks(tick_steps)
            ax2.set_xticklabels([str(int(x)) for x in tick_steps])
        elif len(steps) > 10:
            # 对于中等范围，使用整十刻度
            max_step = steps[-1]
            tick_interval = max(10, max_step // 10)
            tick_steps = np.arange(0, max_step + 1, tick_interval)
            if tick_steps[-1] < max_step:
                tick_steps = np.append(tick_steps, max_step)

            ax2.set_xticks(tick_steps)
            ax2.set_xticklabels([str(int(x)) for x in tick_steps])
        else:
            # 对于小范围，显示所有刻度
            ax2.set_xticks(steps)
            ax2.set_xticklabels([str(int(x)) for x in steps])

        ax2.set_ylabel('PC2/PC1 Ratio', fontsize=12, fontweight='bold')
        ax2.set_title('PC2/PC1 Ratio per Step', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Score Space PCA2 Analysis
        ax3 = axes[2]
        traj = trajectories[0]  # Use first trajectory
        score_pca_proj = score_pca.transform(traj['score'])
        n_points = len(score_pca_proj)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))

        # Plot trajectory
        for j in range(n_points - 1):
            ax3.plot([score_pca_proj[j, 0], score_pca_proj[j+1, 0]],
                   [score_pca_proj[j, 1], score_pca_proj[j+1, 1]],
                   color=colors[j], linewidth=3, alpha=0.9)

        # Mark start and end points
        ax3.scatter(score_pca_proj[0, 0], score_pca_proj[0, 1],
                  c=start_color, s=200, marker='o', label='Start', zorder=5,
                  edgecolors='black', linewidth=2)
        ax3.scatter(score_pca_proj[-1, 0], score_pca_proj[-1, 1],
                  c=end_color, s=200, marker='s', label='End', zorder=5,
                  edgecolors='black', linewidth=2)

        # Add step markers
        for j in range(0, n_points, max(1, n_points//8)):
            ax3.scatter(score_pca_proj[j, 0], score_pca_proj[j, 1],
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
        ax3.set_xlim(x_center - x_display_range/2, x_center + x_display_range/2)
        ax3.set_ylim(y_center - y_display_range/2, y_center + y_display_range/2)

        ax3.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax3.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax3.set_title('Score Space PCA2 Analysis', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10, loc='upper right')
        ax3.grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        # Save the plot
        save_path = os.path.join(results_dir, f'pca2_analysis_{self.method}_{self.model}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\n✓ {self.method} {self.model} PCA2 analysis plot saved to: {save_path}")

        plt.close()

        # 生成局部轨迹图
        self.plot_local_trajectory_windows(trajectories, xt_pca, step_ratios, save_dir)

    def plot_local_trajectory_windows(self, trajectories, xt_pca, step_ratios, save_dir):
        """Generate local trajectory windows at high slope points"""
        print(f"\n📊 Generating local trajectory windows...")

        # 找到高斜率变化点
        high_slope_points = self.find_high_slope_points(step_ratios, num_points=3)
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
            self.plot_local_window(xt_pca_proj, t, ax, f"Local 7-step around t={t}\n(Slope: {np.diff(step_ratios)[t-1]:.4f})")

        plt.tight_layout()

        # 保存局部窗口图
        local_windows_path = os.path.join(results_dir, f'local_windows_{self.method}_{self.model}.png')
        plt.savefig(local_windows_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ {self.method} {self.model} local trajectory windows plot saved to: {local_windows_path}")

        plt.close()

    def generate_analysis_report(self, xt_pca, score_pca, step_ratios, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """Generate analysis report"""
        os.makedirs(save_dir, exist_ok=True)

        report_path = os.path.join(results_dir, f'pca2_analysis_report_{self.method}_{self.model}.txt')

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
            f.write(f"PC1 explained variance: {xt_pca.explained_variance_ratio_[0]:.6f}\n")
            f.write(f"PC2 explained variance: {xt_pca.explained_variance_ratio_[1]:.6f}\n")
            f.write(f"Total explained variance: {np.sum(xt_pca.explained_variance_ratio_):.6f}\n")
            f.write(f"PC1/PC2 ratio: {xt_pca.explained_variance_[0]/xt_pca.explained_variance_[1]:.6f}\n\n")

            f.write("SCORE SPACE PCA2 ANALYSIS:\n")
            f.write("-" * 28 + "\n")
            f.write(f"PC1 explained variance: {score_pca.explained_variance_ratio_[0]:.6f}\n")
            f.write(f"PC2 explained variance: {score_pca.explained_variance_ratio_[1]:.6f}\n")
            f.write(f"Total explained variance: {np.sum(score_pca.explained_variance_ratio_):.6f}\n")
            f.write(f"PC1/PC2 ratio: {score_pca.explained_variance_[0]/score_pca.explained_variance_[1]:.6f}\n\n")

            f.write("PC2/PC1 RATIO PER STEP ANALYSIS:\n")
            f.write("-" * 32 + "\n")
            f.write(f"Mean ratio: {np.mean(step_ratios):.6f}\n")
            f.write(f"Std ratio: {np.std(step_ratios):.6f}\n")
            f.write(f"Min ratio: {np.min(step_ratios):.6f}\n")
            f.write(f"Max ratio: {np.max(step_ratios):.6f}\n")
            f.write(f"Deviation from 1.0: {np.mean(np.abs(step_ratios - 1.0)):.6f}\n\n")

            # 添加高斜率点信息
            high_slope_points = self.find_high_slope_points(step_ratios, num_points=3)
            f.write("HIGH SLOPE POINTS FOR LOCAL WINDOWS:\n")
            f.write("-" * 35 + "\n")
            for i, t in enumerate(high_slope_points):
                slope = np.diff(step_ratios)[t-1] if t > 0 else 0
                f.write(f"Point {i+1}: t={t}, slope={slope:.6f}\n")
            f.write("\n")

            f.write("STEP-BY-STEP RATIOS:\n")
            f.write("-" * 20 + "\n")
            for i, ratio in enumerate(step_ratios):
                f.write(f"Step {i:2d}: {ratio:.6f}\n")

        print(f"\n✓ Analysis report saved to: {report_path}")

        # Print summary to console
        print(f"\n" + "="*80)
        print("DDIM ICLR 1x3 ANALYSIS SUMMARY")
        print("="*80)

        print(f"\nXT Space PCA2:")
        print(f"PC1 explained variance: {xt_pca.explained_variance_ratio_[0]:.6f}")
        print(f"PC2 explained variance: {xt_pca.explained_variance_ratio_[1]:.6f}")
        print(f"PC1/PC2 ratio: {xt_pca.explained_variance_[0]/xt_pca.explained_variance_[1]:.6f}")

        print(f"\nScore Space PCA2:")
        print(f"PC1 explained variance: {score_pca.explained_variance_ratio_[0]:.6f}")
        print(f"PC2 explained variance: {score_pca.explained_variance_ratio_[1]:.6f}")
        print(f"PC1/PC2 ratio: {score_pca.explained_variance_[0]/score_pca.explained_variance_[1]:.6f}")

        print(f"\nPC2/PC1 Ratio per Step:")
        print(f"Mean: {np.mean(step_ratios):.6f}")
        print(f"Std: {np.std(step_ratios):.6f}")
        print(f"Deviation from 1.0: {np.mean(np.abs(step_ratios - 1.0)):.6f}")

        # 打印高斜率点信息
        high_slope_points = self.find_high_slope_points(step_ratios, num_points=3)
        print(f"\nHigh Slope Points for Local Windows:")
        for i, t in enumerate(high_slope_points):
            slope = np.diff(step_ratios)[t-1] if t > 0 else 0
            print(f"  Point {i+1}: t={t}, slope={slope:.6f}")

def main():
    """Main function to run PCA2 analysis"""

    method_name = 'ddim'
    model_type = 'ddpm_ema_cifar10'
    num_inference_steps = 1000
    num_trajectories = 20
    n_samples = 10

    # Initialize analyzer
    analyzer = PCA2Analysis(n_samples=n_samples, num_inference_steps=num_inference_steps, num_trajectories=num_trajectories)

    try:
        # Load pipeline
        print(f"\n{'='*60}")
        print(f"Loading {method_name.upper()} {model_type.upper()} Pipeline")
        print(f"{'='*60}")
        pipe = analyzer.load_pipeline(method_name=method_name, model_type=model_type)

        # Generate trajectories
        print(f"\n{'='*60}")
        print(f"Generating {num_trajectories} Trajectories")
        print(f"{'='*60}")
        trajectories = analyzer.generate_trajectories(pipe, seed=42)

        # Create PCA models
        print(f"\n{'='*60}")
        print(f"Creating PCA Models for {method_name.upper()} {model_type.upper()}")
        print(f"{'='*60}")
        xt_pca, score_pca, xt_data, score_data = analyzer.create_pca_models(trajectories)

        # Calculate PC2/PC1 ratio per step
        print(f"\n{'='*60}")
        print(f"Calculating PC2/PC1 Ratio per Step for {method_name.upper()} {model_type.upper()}")
        print(f"{'='*60}")
        step_ratios = analyzer.calculate_pc2_pc1_ratio_per_step(trajectories, xt_pca)

        # Create ICLR 1x3 analysis plots
        print(f"\n{'='*60}")
        print(f"Creating PCA2 Analysis Plots for {method_name.upper()} {model_type.upper()}")
        print(f"{'='*60}")
        analyzer.plot_iclr_1x3_analysis(trajectories, xt_pca, score_pca, step_ratios)

        # Generate analysis report
        print(f"\n{'='*60}")
        print(f"Generating Analysis Report for {method_name.upper()} {model_type.upper()}")
        print(f"{'='*60}")
        analyzer.generate_analysis_report(xt_pca, score_pca, step_ratios)

        print(f"\n✅ PCA2 analysis completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
