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



# load model and scheduler
pipeline = DiffusionPipeline.from_pretrained(celeba_model_path)

# run pipeline in inference (sample random noise and denoise)
image = pipeline(num_inference_steps=200)["sample"]

# save image
image[0].save(os.path.join(celeba_ldm_save_dir, "ldm_generated_image.png"))


seed = 3

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

image = noise
for t in tqdm.tqdm(scheduler.timesteps):
    # predict noise residual of previous image
    with torch.no_grad():
        residual = unet(image, t)["sample"]

    # compute previous image x_t according to DDIM formula
    prev_image = scheduler.step(residual, t, image, eta=0.0)["prev_sample"]

    # x_t-1 -> x_t
    image = prev_image

# decode image with vae
with torch.no_grad():
    image = vqvae.decode(image).sample

# process image
image_processed = image.permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.clamp(0, 255).cpu().numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])

image_pil.save(os.path.join(celeba_ldm_save_dir, f"generated_image_{seed}.png"))
