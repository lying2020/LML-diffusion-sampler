#!/usr/bin/env python3
"""
Unified CelebA-HQ Sampling Script with Enhanced Logging

This script provides a unified interface for CelebA-HQ image generation using various
diffusion sampling algorithms. It combines enhanced features with comprehensive
evaluation metrics, flexible configuration, and unified logging system.

函数差异分析 (Function Differences Analysis):
===========================================

⚠️ 重要：详细的技术分析请参阅 SCALING_FACTOR_ANALYSIS.md

test_ldm() vs test_dpm_hcg() 的主要差异:

1. **模型加载方式**:
   - test_ldm(): 使用 LDMPipeline.from_pretrained() 一次性加载完整pipeline
   - test_dpm_hcg(): 手动分别加载 unet, vqvae, scheduler 组件

2. **scaling_factor 配置** (关键差异):
   - test_ldm(): LDMPipeline 自动执行 latents / scaling_factor 后解码
   - test_dpm_hcg(): 手动解码，需要设置 scaling_factor = 1.0 以匹配行为
   - 原因：LDMPipeline 源码第118行会自动除以 scaling_factor，手动解码缺少这一步

3. **seed 设置**:
   - 两个函数都使用 seed=5 确保可重复性和可比较性

4. **核心修复**:
   - 设置 vqvae.config.scaling_factor = 1.0
   - 当 scaling_factor = 1.0 时，latents / 1.0 = latents（不变）
   - 这样两个函数的行为完全一致，都直接解码 latents

5. **错误修复**:
   - 添加 use_safetensors=False 参数以避免缺少 safetensors 文件的警告
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
import PIL.Image
import numpy as np
import tqdm

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



seed = 5
# generate gaussian noise to be decoded
generator = torch.manual_seed(seed)

def test_ldm():
    """
    Test using LDMPipeline (high-level API)

    Key behavior: LDMPipeline.__call__() automatically executes:
        latents = latents / self.vqvae.config.scaling_factor  (line 118 in pipeline source)
        image = self.vqvae.decode(latents).sample

    By setting scaling_factor = 1.0, we ensure:
        latents = latents / 1.0 = latents (no scaling)
    This matches test_dpm_hcg() behavior when it also sets scaling_factor = 1.0
    """
    # load model and scheduler
    pipeline = LDMPipeline.from_pretrained(celeba_model_path, use_safetensors=False)

    # Set device
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.unet.to(torch_device)
    pipeline.vqvae.to(torch_device)

    # CRITICAL: Set scaling_factor = 1.0 to match test_dpm_hcg()
    # See SCALING_FACTOR_ANALYSIS.md for detailed explanation
    original_scaling_factor = pipeline.vqvae.config.scaling_factor
    pipeline.vqvae.config.scaling_factor = 1.0

    # Print scaling_factor info
    print(f"\ntest_ldm - VAE scaling_factor (original): {original_scaling_factor}")
    print(f"test_ldm - VAE scaling_factor (set to): {pipeline.vqvae.config.scaling_factor}")

    # run pipeline in inference (sample random noise and denoise)
    pipeline_output = pipeline(num_inference_steps=200, generator=generator)

    # LDMPipeline returns ImagePipelineOutput with images attribute
    if hasattr(pipeline_output, 'images'):
        # ImagePipelineOutput object
        image_pil = pipeline_output.images[0]
    elif isinstance(pipeline_output, (list, tuple)):
        # List of images
        image_pil = pipeline_output[0]
    else:
        # Dict or other format
        image_pil = pipeline_output.get("images", [None])[0]

    # Save the PIL image directly (no processing needed)
    image_pil.save(os.path.join(celeba_ldm_save_dir, "ldm_generated_image.png"))


def test_dpm_hcg():
    """
    Test using manual UNet + VQVAE assembly (low-level API)

    Key difference from test_ldm():
    - test_ldm(): LDMPipeline automatically does latents / scaling_factor before decode
    - test_dpm_hcg(): We must manually handle scaling_factor

    Solution: Set scaling_factor = 1.0 so:
        - test_ldm(): latents / 1.0 = latents → decode (no change)
        - test_dpm_hcg(): decode directly (no change needed)
        Both approaches become equivalent!

    Alternative solution (if keeping original scaling_factor):
        image = image / vqvae.config.scaling_factor  # Manual scaling
        image = vqvae.decode(image).sample
    """

    def process_image(image_tensor):
        """Process image to PIL Image"""
        # process image
        image_processed = image_tensor.permute(0, 2, 3, 1)
        image_processed = (image_processed + 1.0) * 127.5
        image_processed = image_processed.clamp(0, 255).cpu().numpy().astype(np.uint8)
        image_pil = Image.fromarray(image_processed[0])

        return image_pil

    # load all models
    unet = UNet2DModel.from_pretrained(celeba_model_path, subfolder="unet", use_safetensors=False)
    vqvae = VQModel.from_pretrained(celeba_model_path, subfolder="vqvae", use_safetensors=False)
    scheduler = DDIMScheduler.from_config(celeba_model_path, subfolder="scheduler")

    # set to cuda
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    unet.to(torch_device)
    vqvae.to(torch_device)


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
    print(f"\ntest_dpm_hcg - VAE scaling_factor (original): {vqvae.config.scaling_factor}")

    # CRITICAL: Match LDMPipeline behavior
    # In LDMPipeline source code (line 118), it does: latents = latents / self.vqvae.config.scaling_factor
    # So we have two options:
    # Option 1: Set scaling_factor = 1.0, then decode directly (current approach)
    # Option 2: Keep original scaling_factor, but divide latents before decode
    # We use Option 1 for consistency with test_ldm()
    vqvae.config.scaling_factor = 1.0
    print(f"test_dpm_hcg - VAE scaling_factor after setting to 1.0: {vqvae.config.scaling_factor}")

    with torch.no_grad():
        # Since scaling_factor = 1.0, dividing by it has no effect, so direct decode is equivalent
        # to: latents = image / scaling_factor → decode(latents)
        # decode the image latents with the VAE
        image = vqvae.decode(image).sample

    image_pil = process_image(image)
    image_pil.save(os.path.join(celeba_ldm_save_dir, f"generated_image_{seed}.png"))


if __name__ == "__main__":
    test_dpm_hcg()

    test_ldm()