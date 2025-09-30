#!/usr/bin/env python3
"""
Unified CIFAR-10 Sampling Script

This script provides a unified interface for CIFAR-10 image generation using various
diffusion sampling algorithms. It combines the functionality of cifar10.py and
cifar10_updated.py with enhanced features and flexible configuration.
"""

import sys
import time
import torch
import os
import json
import argparse
import glob
from datetime import datetime

sys.path.append(os.getcwd())
from diffusers import DDPMPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_unipc_multistep_lm import UniPCMultistepSchedulerLM
from scheduler.scheduling_ddim_lm import DDIMLMScheduler
from scheduler.scheduling_pndm_lm import PNDMSchedulerLM

import project as project


"""
Examples:
  # Basic usage with default settings
  python cifar10.py --sampler_type dpm_lm --test_num 10

  # Generate with specific output directory
  python cifar10.py --sampler_type ddim --test_num 5 --save_dir ./output/test/cifar10

  # Use different LML parameters
  python cifar10.py --sampler_type dpm_lm --lamb 0.001 --kappa 1e-7 --test_num 20

  # Generate with different data type
  python cifar10.py --sampler_type dpm++ --dtype fp16 --test_num 10
"""

def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="CIFAR-10 sampling script with enhanced features")

    # Basic parameters
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_inference_steps', type=int, default=20)

    # Sampler selection
    parser.add_argument('--sampler_type', type=str, default='pndm',
                        choices=['pndm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc'])
    parser.add_argument('--use_generator', action='store_true', default=True)

    # Output configuration
    parser.add_argument('--save_dir', type=str, default='cifar10')
    parser.add_argument('--model_id', type=str, default='ddpm_ema_cifar10')

    # LML parameters
    parser.add_argument('--lamb', type=float, default=0.0008)
    parser.add_argument('--kappa', type=float, default=1.0e-8)

    # Technical parameters
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp64', 'fp16', 'bf16'])
    parser.add_argument('--device', type=str, default='cuda')

    # Additional options
    parser.add_argument('--save_log', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--run_batch', action='store_true', default=False)
    parser.add_argument('--generate_grid', action='store_true', default=True, help='Generate comparison grid from existing images')

    args = parser.parse_args()

    return args

def get_sampler_description(sampler_type):
    """Get description for different sampler types"""
    descriptions = {
        'ddim': 'Denoising Diffusion Implicit Models - Deterministic sampling',
        'dpm': 'DPM-Solver - High-order solver for diffusion ODEs',
        'dpm++': 'DPM-Solver++ - Improved version with better stability',
        'dpm_lm': 'DPM-Solver with Levenberg-Marquardt Langevin correction',
        'pndm': 'Pseudo Numerical methods for Diffusion Models',
        'unipc': 'Unified Predictor-Corrector framework'
    }
    return descriptions.get(sampler_type, 'Unknown sampler type')

def setup_scheduler(pipe, sampler_type, lamb=0.0008, kappa=1e-8):
    """Setup the appropriate scheduler based on sampler type"""

    if sampler_type == 'pndm':
        pipe.scheduler = PNDMSchedulerLM.from_config(pipe.scheduler.config)
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
        pipe.scheduler = UniPCMultistepSchedulerLM.from_config(pipe.scheduler.config)
        print(f"  Using UniPC scheduler")

    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

def generate_images(pipe, batch_size, num_inference_steps, test_num, start_index, save_dir, sampler_type, use_generator=None):
    """Generate images using the specified pipeline"""

    total_time = 0
    generation_times = []

    print(f"\n{'='*60}")
    print(f"Starting image generation")
    print(f"{'='*60}")
    print(f"Sampler: {sampler_type} - {get_sampler_description(sampler_type)}")
    print(f"Batch size: {batch_size}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Total images: {test_num}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*60}")

    for seed in range(start_index, start_index + test_num):
        print(f"\nGenerating batch {seed - start_index + 1}/{test_num} (seed={seed})")
        batch_start_time = time.time()
        torch.manual_seed(seed)
        if use_generator:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        # Generate images
        with torch.no_grad():
            images = pipe(batch_size=batch_size, num_inference_steps=num_inference_steps, generator=generator).images

        # Save images
        for i, image in enumerate(images):
            filename = f"cifar10_{sampler_type}_inference{num_inference_steps}_seed{seed}_{i}.png"
            filepath = os.path.join(save_dir, filename)
            image.save(filepath)

        batch_time = time.time() - batch_start_time
        generation_times.append(batch_time)
        total_time += batch_time

        print(f"  ✓ Generated {len(images)} images in {batch_time:.3f}s")
        print(f"  ✓ Saved to: {save_dir}")

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
    print(f"{'='*60}")

    return {
        'total_time': total_time,
        'avg_time_per_batch': avg_time,
        'avg_time_per_image': avg_time_per_image,
        'total_images': test_num * batch_size,
        'images_per_second': test_num * batch_size / total_time,
        'generation_times': generation_times
    }

def save_generation_log(save_dir, sampler_type, generation_stats, args):
    """Save generation log to JSON file"""

    log_data = {
        'timestamp': datetime.now().isoformat(),
        'sampler_type': sampler_type,
        'sampler_description': get_sampler_description(sampler_type),
        'parameters': {
            'test_num': args.test_num,
            'start_index': args.start_index,
            'batch_size': args.batch_size,
            'num_inference_steps': args.num_inference_steps,
            'lamb': args.lamb,
            'kappa': args.kappa,
            'dtype': args.dtype,
            'device': args.device
        },
        'generation_stats': generation_stats
    }

    log_filename = f"generation_log_{sampler_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path = os.path.join(save_dir, log_filename)

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"  ✓ Generation log saved to: {log_path}")

def count_generated_images(save_dir):
    """Count the number of generated images in the save directory"""
    try:
        png_files = glob.glob(os.path.join(save_dir, "*.png"))
        return len(png_files)
    except Exception as e:
        print(f"  ⚠️  Warning: Could not count images in {save_dir}: {e}")
        return 0

def run_single_experiment(args, experiment_num, total_experiments, sampler_type, num_inference_steps):
    """Run a single experiment with enhanced logging"""

    print("")
    print("="*50)
    print(f"实验 {experiment_num}/{total_experiments}")
    print(f"Sampler: {sampler_type}")
    print(f"Steps: {num_inference_steps}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    # 记录实验开始时间
    exp_start_time = time.time()

    try:
        # Convert dtype string to torch dtype
        dtype_map = {
            'fp32': torch.float32,
            'fp64': torch.float64,
            'fp16': torch.float16,
            'bf16': torch.bfloat16
        }
        dtype = dtype_map[args.dtype]

        # Setup paths - fix the path handling
        model_id = os.path.join(project.model_dir, args.model_id)

        # Handle save_dir path properly
        save_dir = os.path.join(project.output_dir, args.save_dir, "steps"+'_'+str(num_inference_steps), sampler_type)
        os.makedirs(save_dir, exist_ok=True)

        print(f"🚀 CIFAR-10 Unified Sampling Script")
        print("="*60)
        print(f"Model: {model_id}")
        print(f"Device: {args.device}")
        print(f"Data type: {args.dtype}")
        print(f"Sampler: {sampler_type}")
        print(f"Output: {save_dir}")
        print("="*60)

        # Load pipeline
        print("\n📦 Loading model...")
        pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=False)
        pipe.unet.to(args.device)
        print("  ✓ Model loaded successfully")

        # Setup scheduler
        print(f"\n⚙️ Setting up scheduler...")
        setup_scheduler(pipe, sampler_type, args.lamb, args.kappa)

        # Generate images
        generation_stats = generate_images(
            pipe, args.batch_size, num_inference_steps,
            args.test_num, args.start_index, save_dir, sampler_type, args.use_generator
        )

        # Save generation log if requested
        if args.save_log:
            print(f"\n📝 Saving generation log...")
            save_generation_log(save_dir, sampler_type, generation_stats, args)

        # 计算实验耗时
        exp_end_time = time.time()
        exp_duration = exp_end_time - exp_start_time

        # 统计生成的图片数量
        image_count = count_generated_images(save_dir)

        print(f"\n✅ 实验完成! 耗时: {exp_duration:.1f}秒")
        print(f"生成图片数量: {image_count}")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            'success': True,
            'duration': exp_duration,
            'image_count': image_count,
            'generation_stats': generation_stats,
            'save_dir': save_dir
        }

    except Exception as e:
        exp_end_time = time.time()
        exp_duration = exp_end_time - exp_start_time

        print(f"\n❌ 实验失败! 耗时: {exp_duration:.1f}秒")
        print(f"错误信息: {str(e)}")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            'success': False,
            'duration': exp_duration,
            'error': str(e),
            'sampler_type': sampler_type,
            'num_inference_steps': num_inference_steps
        }

def generate_experiment_summary(experiment_results, start_time, end_time, args):
    """Generate comprehensive experiment summary report"""

    summary_file = os.path.join(project.output_dir, args.save_dir, "experiment_summary.txt")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    successful_experiments = [r for r in experiment_results if r['success']]
    failed_experiments = [r for r in experiment_results if not r['success']]

    total_images = sum(r.get('image_count', 0) for r in successful_experiments)
    total_duration = sum(r.get('duration', 0) for r in experiment_results)

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("CIFAR-10 实验总结报告\n")
        f.write("="*50 + "\n\n")
        f.write(f"运行时间: {start_time} 到 {end_time}\n")
        f.write(f"总实验数: {len(experiment_results)}\n")
        f.write(f"成功实验: {len(successful_experiments)}\n")
        f.write(f"失败实验: {len(failed_experiments)}\n")
        f.write(f"总生成图片: {total_images}\n")
        f.write(f"总耗时: {total_duration:.1f}秒\n\n")

        f.write("参数配置:\n")
        f.write(f"- Batch Size: {args.batch_size}\n")
        f.write(f"- Test Num: {args.test_num}\n")
        f.write(f"- Device: {args.device}\n")
        f.write(f"- Data Type: {args.dtype}\n")
        f.write(f"- Lambda: {args.lamb}\n")
        f.write(f"- Kappa: {args.kappa}\n\n")

        f.write("各实验详情:\n")
        for result in experiment_results:
            if result['success']:
                f.write(f"- {result.get('sampler_type', 'unknown')} ({result.get('num_inference_steps', 'unknown')} steps): {result.get('image_count', 0)} 张图片, 耗时 {result.get('duration', 0):.1f}秒\n")
            else:
                f.write(f"- {result.get('sampler_type', 'unknown')} ({result.get('num_inference_steps', 'unknown')} steps): 失败 - {result.get('error', 'Unknown error')}\n")

    print(f"\n📊 实验总结报告已保存到: {summary_file}")

    # 保存失败实验日志
    if failed_experiments:
        failed_log_file = os.path.join(project.output_dir, args.save_dir, "failed_experiments.log")
        with open(failed_log_file, 'w', encoding='utf-8') as f:
            for result in failed_experiments:
                f.write(f"{result.get('sampler_type', 'unknown')},{result.get('num_inference_steps', 'unknown')},{result.get('error', 'Unknown error')}\n")
        print(f"⚠️  有 {len(failed_experiments)} 个实验失败，详情请查看: {failed_log_file}")
    else:
        print("🎉 所有实验都成功完成!")

def generate_comparison_grid(save_dir, sampler_types, num_inference_steps=20, test_num=6, batch_size=1):
    """
    生成类似截图的对比图组，6行多列展示不同采样方法的结果

    Args:
        save_dir: 保存目录
        sampler_types: 采样器类型列表
        num_inference_steps: 推理步数
        test_num: 测试数量（行数）
        batch_size: 批次大小
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    from PIL import Image
    import glob

    print(f"\n🎨 生成对比图组...")
    print(f"采样器: {sampler_types}")
    print(f"行数: {test_num}, 列数: {len(sampler_types)}")

    # 创建图像网格
    fig = plt.figure(figsize=(len(sampler_types) * 2.5, test_num * 2.5))
    gs = GridSpec(test_num, len(sampler_types), figure=fig,
                  hspace=0.1, wspace=0.05,
                  left=0.05, right=0.95, top=0.95, bottom=0.05)

    # 方法名称映射
    method_names = {
        'ddim': 'DDIM',
        'pndm': 'PNDM',
        'dpm': 'DPM-Solver',
        'dpm++': 'DPM-Solver++',
        'unipc': 'UniPC',
        'dpm_lm': 'LML (Ours)'
    }

    # 为每个采样器生成图像
    all_images = {}

    for col, sampler_type in enumerate(sampler_types):
        print(f"  生成 {sampler_type} 图像...")

        # 设置路径
        sampler_dir = os.path.join(save_dir, "steps" + '_' + str(num_inference_steps), sampler_type)

        # 查找该采样器生成的图像
        image_files = glob.glob(os.path.join(sampler_dir, "*.png"))
        if not image_files:
            print(f"  ⚠️  未找到 {sampler_type} 的图像文件")
            continue

        # 按文件名排序，取前test_num个
        image_files.sort()
        selected_images = image_files[:test_num]
        all_images[sampler_type] = selected_images

    # 绘制图像网格
    for row in range(test_num):
        for col, sampler_type in enumerate(sampler_types):
            ax = fig.add_subplot(gs[row, col])

            if sampler_type in all_images and row < len(all_images[sampler_type]):
                # 加载并显示图像
                img_path = all_images[sampler_type][row]
                try:
                    img = Image.open(img_path)
                    ax.imshow(img)
                except Exception as e:
                    print(f"  ⚠️  无法加载图像 {img_path}: {e}")
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            else:
                # 显示占位符
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='gray')

            # 设置坐标轴
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

            # 添加列标题（只在第一行）
            if row == 0:
                method_name = method_names.get(sampler_type, sampler_type)
                ax.set_title(method_name, fontsize=12, fontweight='bold', pad=10)

            # 添加行标签（只在第一列）
            if col == 0:
                ax.text(-0.15, 0.5, f'Row {row+1}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, rotation=90)

    # 添加分隔线（在LML列后）
    if 'dpm_lm' in sampler_types:
        lml_index = sampler_types.index('dpm_lm')
        if lml_index < len(sampler_types) - 1:
            # 在LML列后添加垂直分隔线
            for row in range(test_num):
                ax = fig.add_subplot(gs[row, lml_index])
                # 添加右侧边框
                ax.add_patch(patches.Rectangle((0.95, 0), 0.05, 1,
                                             transform=ax.transAxes,
                                             facecolor='black', alpha=0.3))

    # 设置整体标题
    fig.suptitle(f'CIFAR-10 Generation Comparison (Steps: {num_inference_steps})',
                 fontsize=16, fontweight='bold', y=0.98)

    # 保存图像
    output_path = os.path.join(save_dir, f'comparison_grid_steps{num_inference_steps}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ 对比图组已保存到: {output_path}")
    return output_path

def generate_comparison_grid_from_existing(save_dir, num_inference_steps=20):
    """
    从已存在的图像文件生成对比图组

    Args:
        save_dir: 保存目录
        num_inference_steps: 推理步数
    """
    # 定义采样器类型
    sampler_types = ['ddim', 'pndm', 'dpm', 'dpm++', 'unipc', 'dpm_lm']

    # 检查哪些采样器有图像
    available_samplers = []
    for sampler in sampler_types:
        sampler_dir = os.path.join(save_dir, "steps" + '_' + str(num_inference_steps), sampler)
        if os.path.exists(sampler_dir) and glob.glob(os.path.join(sampler_dir, "*.png")):
            available_samplers.append(sampler)

    if not available_samplers:
        print("❌ 未找到任何采样器的图像文件")
        return None

    print(f"找到 {len(available_samplers)} 个采样器的图像: {available_samplers}")

    # 生成对比图组
    return generate_comparison_grid(save_dir, available_samplers, num_inference_steps)

def main(args):
    """Main function for single experiment"""
    print(args)

    # Convert dtype string to torch dtype
    dtype_map = {
        'fp32': torch.float32,
        'fp64': torch.float64,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]

    # Setup paths - fix the path handling
    model_id = os.path.join(project.model_dir, args.model_id)

    # Handle save_dir path properly
    save_dir = os.path.join(project.output_dir, args.save_dir, "steps"+'_'+str(args.num_inference_steps), args.sampler_type)
    os.makedirs(save_dir, exist_ok=True)

    print("🚀 CIFAR-10 Unified Sampling Script")
    print("="*60)
    print(f"Model: {model_id}")
    print(f"Device: {args.device}")
    print(f"Data type: {args.dtype}")
    print(f"Sampler: {args.sampler_type}")
    print(f"Output: {save_dir}")
    print("="*60)

    # Load pipeline
    print("\n📦 Loading model...")
    pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=False)
    pipe.unet.to(args.device)
    print("  ✓ Model loaded successfully")

    # Setup scheduler
    print(f"\n⚙️ Setting up scheduler...")
    setup_scheduler(pipe, args.sampler_type, args.lamb, args.kappa)

    # Generate images
    generation_stats = generate_images(
        pipe, args.batch_size, args.num_inference_steps,
        args.test_num, args.start_index, save_dir, args.sampler_type, args.use_generator
    )

    # Save generation log if requested
    if args.save_log:
        print(f"\n📝 Saving generation log...")
        save_generation_log(save_dir, args.sampler_type, generation_stats, args)

    print(f"\n✅ Generation completed successfully!")
    print(f"   Generated {generation_stats['total_images']} images")
    print(f"   Total time: {generation_stats['total_time']:.2f}s")
    print(f"   Average time per image: {generation_stats['avg_time_per_image']:.3f}s")


if __name__ == '__main__':
    args = parse_args()

    # 检查是否运行批量实验
    if hasattr(args, 'run_batch') and args.run_batch:
        # 批量实验模式
        SAMPLER_TYPES = ["pndm", "ddim", "dpm++", "dpm", "unipc"]
        INFERENCE_STEPS = [5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 80]

        print("="*50)
        print("CIFAR-10 实验批量运行开始")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)

        total_experiments = len(SAMPLER_TYPES) * len(INFERENCE_STEPS)
        experiment_results = []
        start_time = datetime.now()

        experiment_num = 0
        for i in range(len(INFERENCE_STEPS)):
            for j in range(len(SAMPLER_TYPES)):
                experiment_num += 1
                num_inference_steps = INFERENCE_STEPS[i]
                sampler_type = SAMPLER_TYPES[j]

                # 更新args
                args.num_inference_steps = num_inference_steps
                args.sampler_type = sampler_type

                result = run_single_experiment(args, experiment_num, total_experiments, sampler_type, num_inference_steps)
                experiment_results.append(result)

        end_time = datetime.now()

        print("\n" + "="*50)
        print("CIFAR-10 实验批量运行完成")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总实验数: {total_experiments}")
        print("="*50)

        # 生成实验总结报告
        print("\n生成实验总结报告...")
        generate_experiment_summary(experiment_results, start_time, end_time, args)

        print(f"\n实验完成! 结果保存在: {os.path.join(project.output_dir, args.save_dir)}")
    elif args.generate_grid:
        # 生成对比图组模式
        print("\n🎨 生成对比图组...")
        output_path = generate_comparison_grid_from_existing(
            os.path.join(project.output_dir, args.save_dir), args.num_inference_steps
        )
        if output_path:
            print(f"✅ 对比图组已生成: {output_path}")
        else:
            print("❌ 对比图组生成失败")
    else:
        # 单个实验模式（保持原有逻辑）
        SAMPLER_TYPES=["pndm", "ddim", "dpm++", "dpm", "unipc"]
        # INFERENCE_STEPS=[5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 80]
        INFERENCE_STEPS=[5, 6]

        for i in range(len(INFERENCE_STEPS)):
            for j in range(len(SAMPLER_TYPES)):
                args.num_inference_steps = INFERENCE_STEPS[i]
                args.sampler_type = SAMPLER_TYPES[j]
                main(args)
