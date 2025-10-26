#!/usr/bin/env python3
"""
Unified CelebA-HQ Sampling Script with Enhanced Logging

This script provides a unified interface for CelebA-HQ image generation using various
diffusion sampling algorithms. It combines enhanced features with comprehensive
evaluation metrics, flexible configuration, and unified logging system.
"""

import sys
import time
import torch
import os
import json
import argparse
import glob
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.getcwd())
from diffusers import DiffusionPipeline
from diffusers import UNet2DModel, DDIMScheduler, VQModel
from diffusers import LDMPipeline, DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler
from scheduler.scheduling_dpmsolver_multistep_hcg import DPMSolverMultistepHCGScheduler

import project as project

celeba_model_path = "/home/liying/Documents/ldm-celebahq-256/"
celeba_ldm_save_dir = os.path.join(project.output_dir, "celeba_ldm")
os.makedirs(celeba_ldm_save_dir, exist_ok=True)

import PIL.Image
import numpy as np
import tqdm


def pil_to_tensor(image_pil):
    """Convert PIL Image to tensor format expected by process_image"""
    # Convert PIL to numpy array
    image_array = np.array(image_pil).astype(np.float32)

    # Convert from [0, 255] to [-1, 1] range
    image_normalized = (image_array / 127.5) - 1.0

    # Add batch dimension and convert to (B, C, H, W) format
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)

    return image_tensor

def process_image(image_tensor):
    """Process image to PIL Image"""
    # process image
    image_processed = image_tensor.permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.clamp(0, 255).cpu().numpy().astype(np.uint8)
    image_pil = Image.fromarray(image_processed[0])

    return image_pil

# load model and scheduler
# IMPORTANT: Use LDMPipeline instead of DiffusionPipeline for this model
pipeline = LDMPipeline.from_pretrained(celeba_model_path)

# # run pipeline in inference (sample random noise and denoise)
# pipeline_output = pipeline(num_inference_steps=200)

# # LDMPipeline returns ImagePipelineOutput with images attribute
# if hasattr(pipeline_output, 'images'):
#     # ImagePipelineOutput object
#     image_pil = pipeline_output.images[0]
# elif isinstance(pipeline_output, (list, tuple)):
#     # List of images
#     image_pil = pipeline_output[0]
# else:
#     # Dict or other format
#     image_pil = pipeline_output.get("images", [None])[0]

# # Check image data statistics
# image_array = np.array(image_pil)
# print(f"Pipeline output - Image shape: {image_array.shape}")
# print(f"Pipeline output - Min value: {image_array.min()}, Max value: {image_array.max()}")
# print(f"Pipeline output - Mean value: {image_array.mean():.3f}, Std: {image_array.std():.3f}")

# # Save the PIL image directly (no processing needed)
# image_pil.save(os.path.join(celeba_ldm_save_dir, "ldm_generated_image.png"))


seed = 5

# load all models
unet = UNet2DModel.from_pretrained(celeba_model_path, subfolder="unet")
vqvae = VQModel.from_pretrained(celeba_model_path, subfolder="vqvae")
scheduler = DDIMScheduler.from_config(celeba_model_path, subfolder="scheduler")

# set to cuda
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

unet.to(torch_device)
vqvae.to(torch_device)

# generate gaussian noise to be decoded
generator = torch.manual_seed(seed)
noise = torch.randn(
    (1, unet.in_channels, unet.sample_size, unet.sample_size),
    generator=generator,
).to(torch_device)

# set inference steps for DDIM
scheduler.set_timesteps(num_inference_steps=200)

# Note: init_noise_sigma is 1.0 for DDIMScheduler, so no scaling needed here
image = noise
for t in tqdm.tqdm(scheduler.timesteps):
    # predict noise residual of previous image
    with torch.no_grad():
        # CRITICAL: scale model input as done in LDMPipeline line 111
        image = scheduler.scale_model_input(image, t)
        residual = unet(image, t)["sample"]

    # compute previous image x_t according to DDIM formula
    prev_image = scheduler.step(residual, t, image, eta=0.0)["prev_sample"]

    # x_t-1 -> x_t
    image = prev_image

# decode image with vae (with proper scaling as in LDMPipeline)
print(f"\nBefore VAE decode - Latent stats: shape={image.shape}, min={image.min():.3f}, max={image.max():.3f}, mean={image.mean():.3f}")
print(f"VAE scaling_factor: {vqvae.config.scaling_factor}")

with torch.no_grad():
    # CRITICAL STEP: Adjust latents with inverse of vae scale (line 118 in pipeline)
    # image = image / vqvae.config.scaling_factor
    # print(f"After scaling - Latent stats: min={image.min():.3f}, max={image.max():.3f}, mean={image.mean():.3f}")

    # decode the image latents with the VAE
    image = vqvae.decode(image).sample

image_pil = process_image(image)
image_pil.save(os.path.join(celeba_ldm_save_dir, f"generated_image_{seed}.png"))
