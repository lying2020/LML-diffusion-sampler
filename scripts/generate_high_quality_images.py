#!/usr/bin/env python3
"""
High Quality CIFAR-10 Image Generation Script

This script generates high-quality CIFAR-10 images with better visualization
and quality analysis capabilities.
"""

import sys
import time
import torch
import os
import json
import argparse
from datetime import datetime
import numpy as np
from PIL import Image, ImageEnhance

sys.path.append(os.getcwd())
from diffusers import DDPMPipeline, DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler

import project as project

def analyze_image_quality(img_array):
    """Analyze image quality metrics"""
    # 基本统计
    mean_val = img_array.mean()
    std_val = img_array.std()
    unique_colors = len(np.unique(img_array))
    
    # 边缘检测
    grad_x = np.abs(np.diff(img_array, axis=1))
    grad_y = np.abs(np.diff(img_array, axis=0))
    edge_strength = (grad_x.mean() + grad_y.mean()) / 2
    
    # 对比度
    contrast = img_array.std()
    
    # 亮度分布
    brightness = img_array.mean()
    
    # 颜色丰富度
    color_richness = unique_colors / (32 * 32 * 3) * 100
    
    return {
        'mean_brightness': float(mean_val),
        'std_deviation': float(std_val),
        'unique_colors': int(unique_colors),
        'edge_strength': float(edge_strength),
        'contrast': float(contrast),
        'color_richness': float(color_richness)
    }

def enhance_image(img, enhancement_factor=1.2):
    """Enhance image quality"""
    # 增强对比度
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(enhancement_factor)
    
    # 增强锐度
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(enhancement_factor)
    
    return img

def create_visualization_grid(images, grid_size=(4, 4), image_size=(128, 128)):
    """Create a grid visualization of images"""
    if len(images) == 0:
        return None
    
    # 计算网格大小
    rows, cols = grid_size
    if len(images) < rows * cols:
        rows = (len(images) + cols - 1) // cols
    
    # 创建网格图片
    grid_width = cols * image_size[0]
    grid_height = rows * image_size[1]
    grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for i, img in enumerate(images[:rows * cols]):
        row = i // cols
        col = i % cols
        
        # 调整图片大小
        resized_img = img.resize(image_size, Image.LANCZOS)
        
        # 放置到网格中
        x = col * image_size[0]
        y = row * image_size[1]
        grid_img.paste(resized_img, (x, y))
    
    return grid_img

def generate_high_quality_images(pipe, batch_size, num_inference_steps, test_num, start_index, save_dir, sampler_type, enhancement=True):
    """Generate high-quality images with analysis"""
    
    total_time = 0
    generation_times = []
    quality_metrics = []
    all_images = []
    
    print(f"\n{'='*60}")
    print(f"Starting HIGH-QUALITY image generation")
    print(f"{'='*60}")
    print(f"Sampler: {sampler_type}")
    print(f"Batch size: {batch_size}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Total images: {test_num}")
    print(f"Enhancement: {'ON' if enhancement else 'OFF'}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*60}")
    
    for seed in range(start_index, start_index + test_num):
        print(f"\nGenerating batch {seed - start_index + 1}/{test_num} (seed={seed})")
        batch_start_time = time.time()
        torch.manual_seed(seed)

        # Generate images
        with torch.no_grad():
            images = pipe(batch_size=batch_size, num_inference_steps=num_inference_steps).images

        # Process and enhance images
        processed_images = []
        batch_metrics = []
        
        for i, image in enumerate(images):
            # 分析原始图片质量
            img_array = np.array(image)
            quality = analyze_image_quality(img_array)
            batch_metrics.append(quality)
            
            # 增强图片质量（如果启用）
            if enhancement:
                enhanced_image = enhance_image(image, enhancement_factor=1.1)
            else:
                enhanced_image = image
            
            # 保存原始图片
            original_filename = f"cifar10_{sampler_type}_inference{num_inference_steps}_seed{seed}_{i}.png"
            original_filepath = os.path.join(save_dir, original_filename)
            image.save(original_filepath)
            
            # 保存增强图片
            enhanced_filename = f"cifar10_{sampler_type}_enhanced_inference{num_inference_steps}_seed{seed}_{i}.png"
            enhanced_filepath = os.path.join(save_dir, enhanced_filename)
            enhanced_image.save(enhanced_filepath)
            
            processed_images.append(enhanced_image)
            all_images.append(enhanced_image)
            
            print(f"  ✓ Generated image {i+1}: {original_filename}")
            print(f"    Quality: brightness={quality['mean_brightness']:.1f}, contrast={quality['contrast']:.1f}, colors={quality['unique_colors']}")
        
        batch_time = time.time() - batch_start_time
        generation_times.append(batch_time)
        total_time += batch_time
        quality_metrics.extend(batch_metrics)
        
        print(f"  ✓ Batch completed in {batch_time:.3f}s")
    
    # 创建可视化网格
    if all_images:
        print(f"\n📊 Creating visualization grid...")
        grid_img = create_visualization_grid(all_images, grid_size=(4, 4), image_size=(128, 128))
        if grid_img:
            grid_path = os.path.join(save_dir, f"cifar10_{sampler_type}_grid_visualization.png")
            grid_img.save(grid_path)
            print(f"  ✓ Grid visualization saved: {grid_path}")
    
    # 计算质量统计
    avg_quality = {
        'mean_brightness': np.mean([m['mean_brightness'] for m in quality_metrics]),
        'std_deviation': np.mean([m['std_deviation'] for m in quality_metrics]),
        'unique_colors': np.mean([m['unique_colors'] for m in quality_metrics]),
        'edge_strength': np.mean([m['edge_strength'] for m in quality_metrics]),
        'contrast': np.mean([m['contrast'] for m in quality_metrics]),
        'color_richness': np.mean([m['color_richness'] for m in quality_metrics])
    }
    
    # Print summary
    avg_time = total_time / test_num
    avg_time_per_image = avg_time / batch_size
    
    print(f"\n{'='*60}")
    print(f"GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per batch: {avg_time:.3f}s")
    print(f"Average time per image: {avg_time_per_image:.3f}s")
    print(f"Total images generated: {test_num * batch_size}")
    print(f"Images per second: {test_num * batch_size / total_time:.2f}")
    print(f"\nQUALITY METRICS:")
    print(f"Average brightness: {avg_quality['mean_brightness']:.2f}")
    print(f"Average contrast: {avg_quality['contrast']:.2f}")
    print(f"Average unique colors: {avg_quality['unique_colors']:.0f}")
    print(f"Average edge strength: {avg_quality['edge_strength']:.2f}")
    print(f"Color richness: {avg_quality['color_richness']:.2f}%")
    print(f"{'='*60}")
    
    return {
        'total_time': total_time,
        'avg_time_per_batch': avg_time,
        'avg_time_per_image': avg_time_per_image,
        'total_images': test_num * batch_size,
        'images_per_second': test_num * batch_size / total_time,
        'generation_times': generation_times,
        'quality_metrics': quality_metrics,
        'avg_quality': avg_quality
    }

def main():
    parser = argparse.ArgumentParser(
        description="High Quality CIFAR-10 image generation with analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Basic parameters
    parser.add_argument('--test_num', type=int, default=4,
                        help='Number of test batches to generate (default: 4)')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Starting seed index (default: 0)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of images per batch (default: 4)')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of denoising steps (default: 50)')
    
    # Sampler selection
    parser.add_argument('--sampler_type', type=str, default='dpm_lm',
                        choices=['pndm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc'],
                        help='Type of sampler to use (default: dpm_lm)')
    
    # Output configuration
    parser.add_argument('--save_dir', type=str, default='./output/test/high_quality',
                        help='Directory to save generated images (default: ./output/test/high_quality)')
    parser.add_argument('--model_id', type=str, default='ddpm_ema_cifar10',
                        help='Model directory name (default: ddpm_ema_cifar10)')
    
    # LML parameters
    parser.add_argument('--lamb', type=float, default=0.0008,
                        help='Lambda parameter for LML correction (default: 0.0008)')
    parser.add_argument('--kappa', type=float, default=1.0e-8,
                        help='Kappa parameter for EMA smoothing (default: 1.0e-8)')
    
    # Technical parameters
    parser.add_argument('--dtype', type=str, default='fp32',
                        choices=['fp32', 'fp64', 'fp16', 'bf16'],
                        help='Data type for computation (default: fp32)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation (default: cuda)')
    
    # Quality options
    parser.add_argument('--no_enhancement', action='store_true',
                        help='Disable image enhancement')
    parser.add_argument('--save_log', action='store_true',
                        help='Save generation log to JSON file')

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        'fp32': torch.float32,
        'fp64': torch.float64,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]

    # Setup paths
    model_id = os.path.join(project.model_dir, args.model_id)
    
    # Handle save_dir path properly
    if args.save_dir.startswith('./output/'):
        save_dir = os.path.join(project.output_dir, args.save_dir[9:], args.sampler_type)
    else:
        save_dir = os.path.join(project.output_dir, args.save_dir, args.sampler_type)
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("🚀 High Quality CIFAR-10 Image Generation")
    print("="*60)
    print(f"Model: {model_id}")
    print(f"Device: {args.device}")
    print(f"Data type: {args.dtype}")
    print(f"Sampler: {args.sampler_type}")
    print(f"Output: {save_dir}")
    print(f"Enhancement: {'OFF' if args.no_enhancement else 'ON'}")
    print("="*60)

    try:
        # Load pipeline
        print("\n📦 Loading model...")
        pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=False)
        pipe.unet.to(args.device)
        print("  ✓ Model loaded successfully")

        # Setup scheduler (simplified version)
        print(f"\n⚙️ Setting up scheduler...")
        if args.sampler_type == 'ddim':
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif args.sampler_type == 'dpm_lm':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lamb = args.lamb
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = args.kappa
        # Add other schedulers as needed
        print(f"  ✓ Scheduler configured")

        # Generate images
        generation_stats = generate_high_quality_images(
            pipe, args.batch_size, args.num_inference_steps, 
            args.test_num, args.start_index, save_dir, args.sampler_type,
            enhancement=not args.no_enhancement
        )

        # Save generation log if requested
        if args.save_log:
            print(f"\n📝 Saving generation log...")
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'sampler_type': args.sampler_type,
                'parameters': vars(args),
                'generation_stats': generation_stats
            }
            
            log_filename = f"high_quality_generation_log_{args.sampler_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            log_path = os.path.join(save_dir, log_filename)
            
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"  ✓ Generation log saved to: {log_path}")

        print(f"\n✅ High-quality generation completed successfully!")
        print(f"   Generated {generation_stats['total_images']} images")
        print(f"   Total time: {generation_stats['total_time']:.2f}s")
        print(f"   Average time per image: {generation_stats['avg_time_per_image']:.3f}s")
        print(f"   Average quality score: {generation_stats['avg_quality']['color_richness']:.2f}%")

    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
