#!/usr/bin/env python3
"""
测试不同的 scaling_factor 值对图像生成的影响
用于确定正确的 scaling_factor 配置

核心理解：
- scaling_factor 的作用：将 latent 空间缩放到适合 UNet 训练的数值范围
- 训练时：如果使用了 scaling_factor，编码后会乘以它：latents * scaling_factor
- 推理时：解码前需要除以它恢复原始空间：latents / scaling_factor

关键发现：
- 如果训练时没有使用 scaling_factor（或 scaling_factor = 1.0）：
  - VAE Decoder 期望接收原始 latent 空间的数值
  - 推理时应该直接解码，不需要除以 scaling_factor
  - 因此设置 scaling_factor = 1.0 是正确的

- 如果训练时使用了 scaling_factor（如 0.18215）：
  - VAE Decoder 期望接收除以 scaling_factor 后的数值（原始空间）
  - 推理时需要除以 scaling_factor 恢复到原始空间，然后解码
"""

import torch
import os
from diffusers import LDMPipeline, UNet2DModel, DDIMScheduler, VQModel
from PIL import Image
import numpy as np
import tqdm

celeba_model_path = "/home/liying/Documents/ldm-celebahq-256/"
output_dir = "/home/user/GeoDiff_CVPR/output/celeba_ldm/scaling_factor_test"
os.makedirs(output_dir, exist_ok=True)

seed = 5
generator = torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

def test_pipeline_with_scaling_factor(scaling_factor, num_steps=200):
    """使用 LDMPipeline 测试不同的 scaling_factor"""
    print(f"\n{'='*60}")
    print(f"测试 LDMPipeline with scaling_factor = {scaling_factor}")
    print(f"{'='*60}")

    pipeline = LDMPipeline.from_pretrained(celeba_model_path, use_safetensors=False)
    pipeline.unet.to(device)
    pipeline.vqvae.to(device)

    # 设置 scaling_factor
    original_sf = pipeline.vqvae.config.scaling_factor
    pipeline.vqvae.config.scaling_factor = scaling_factor

    print(f"原始 scaling_factor: {original_sf}")
    print(f"设置后 scaling_factor: {pipeline.vqvae.config.scaling_factor}")

    # 生成图像
    output = pipeline(num_inference_steps=num_steps, generator=generator)
    image = output.images[0]

    # 保存
    filename = f"pipeline_sf_{scaling_factor}.png"
    image.save(os.path.join(output_dir, filename))
    print(f"已保存: {filename}")

    # 计算图像统计信息
    img_array = np.array(image)
    print(f"图像统计: min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.2f}, std={img_array.std():.2f}")

    return image

def test_manual_with_scaling_factor(scaling_factor, num_steps=200):
    """手动解码测试不同的 scaling_factor"""
    print(f"\n{'='*60}")
    print(f"测试手动解码 with scaling_factor = {scaling_factor}")
    print(f"{'='*60}")

    # 加载模型
    unet = UNet2DModel.from_pretrained(celeba_model_path, subfolder="unet", use_safetensors=False)
    vqvae = VQModel.from_pretrained(celeba_model_path, subfolder="vqvae", use_safetensors=False)
    scheduler = DDIMScheduler.from_config(celeba_model_path, subfolder="scheduler")

    unet.to(device)
    vqvae.to(device)

    original_sf = vqvae.config.scaling_factor
    vqvae.config.scaling_factor = scaling_factor

    print(f"原始 scaling_factor: {original_sf}")
    print(f"设置后 scaling_factor: {vqvae.config.scaling_factor}")

    # 生成噪声
    noise = torch.randn(
        (1, unet.in_channels, unet.sample_size, unet.sample_size),
        generator=generator,
    ).to(device)

    scheduler.set_timesteps(num_inference_steps=num_steps)

    # UNet 去噪
    latents = noise
    for t in tqdm.tqdm(scheduler.timesteps, desc="Denoising"):
        with torch.no_grad():
            latents = scheduler.scale_model_input(latents, t)
            residual = unet(latents, t)["sample"]
            latents = scheduler.step(residual, t, latents, eta=0.0)["prev_sample"]

    print(f"\n去噪后 latents 统计: min={latents.min():.3f}, max={latents.max():.3f}, mean={latents.mean():.3f}")

    # 解码
    with torch.no_grad():
        # 根据 scaling_factor 决定是否缩放
        # 注意：这里模拟 LDMPipeline 的行为（在解码前除以 scaling_factor）
        # 如果模型训练时没有使用 scaling_factor，则 scaling_factor=1.0 时效果最好
        if scaling_factor != 1.0:
            print(f"手动执行缩放: latents = latents / {scaling_factor}")
            print(f"  (模拟 LDMPipeline 的行为，将 latents 从缩放空间恢复到原始空间)")
            scaled_latents = latents / scaling_factor
            print(f"缩放后 latents 统计: min={scaled_latents.min():.3f}, max={scaled_latents.max():.3f}")
            decoded = vqvae.decode(scaled_latents).sample
        else:
            print(f"scaling_factor = 1.0, 直接解码（无需缩放）")
            print(f"  (说明模型训练时没有使用 scaling_factor，VAE Decoder 期望原始 latent 空间)")
            decoded = vqvae.decode(latents).sample

    # 后处理
    image_processed = decoded.permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.clamp(0, 255).cpu().numpy().astype(np.uint8)
    image_pil = Image.fromarray(image_processed[0])

    # 保存
    filename = f"manual_sf_{scaling_factor}.png"
    image_pil.save(os.path.join(output_dir, filename))
    print(f"已保存: {filename}")

    # 计算图像统计信息
    img_array = np.array(image_pil)
    print(f"图像统计: min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.2f}, std={img_array.std():.2f}")

    return image_pil

if __name__ == "__main__":
    # 测试不同的 scaling_factor 值
    test_values = [0.18215, 1.0]

    print("="*60)
    print("开始测试不同的 scaling_factor 值")
    print("="*60)

    # 测试 pipeline
    for sf in test_values:
        try:
            test_pipeline_with_scaling_factor(sf, num_steps=50)  # 使用较少步数以加快测试
        except Exception as e:
            print(f"Pipeline 测试失败 (scaling_factor={sf}): {e}")
            import traceback
            traceback.print_exc()

    # 测试手动解码
    for sf in test_values:
        try:
            test_manual_with_scaling_factor(sf, num_steps=50)
        except Exception as e:
            print(f"手动解码测试失败 (scaling_factor={sf}): {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("测试完成！请检查输出目录中的图像质量:")
    print(f"{output_dir}")
    print("对比不同 scaling_factor 的图像，确定哪个值产生最佳效果")
    print(f"{'='*60}")
