import os
import json
import glob
import textwrap
from datetime import datetime
from pathlib import Path

# 项目路径配置
project_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_dir, 'model')
data_dir = os.path.join(project_dir, 'data')
output_dir = os.path.join(project_dir, 'output')
output_test_dir = os.path.join(output_dir, 'test')

# Import logger from utils (set default output_dir first)
from utils.logger import (
    ProjectLogger, get_logger, setup_logging,
    log_experiment_start, log_experiment_end, log_progress,
    log_error, log_success, log_warning,
    info, warning, error, debug, success, progress,
    set_default_output_dir
)
# Set default output directory for logs
set_default_output_dir(output_dir)

# Import logging module for compatibility
import logging



def get_sampler_description(sampler_type):
    """Get description for different sampler types"""
    descriptions = {
        'ddim': 'Denoising Diffusion Implicit Models - Deterministic sampling',
        'ddim_lm': 'DDIM with Levenberg-Marquardt Langevin correction',
        'dpm': 'DPM-Solver - High-order solver for diffusion ODEs',
        'dpm++': 'DPM-Solver++ - Improved version with better stability',
        'dpm_lm': 'DPM-Solver with Levenberg-Marquardt Langevin correction',
        'pndm': 'Pseudo Numerical methods for Diffusion Models',
        'unipc': 'Unified Predictor-Corrector framework',
        'dpm_hcg': 'DPM-Solver with Hessian-Free HVP correction using CG'
    }
    return descriptions.get(sampler_type, 'Unknown sampler type')

def find_image_files(directory, extensions=None):
    """
    Find image files in a directory, supporting multiple formats.

    Args:
        directory: Directory path to search
        extensions: List of extensions to search for (default: ['.png', '.jpg', '.jpeg'])

    Returns:
        List of image file paths, sorted by filename
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg']

    image_files = []
    for ext in extensions:
        # Support both lowercase and uppercase extensions
        pattern_lower = os.path.join(directory, f"*{ext.lower()}")
        pattern_upper = os.path.join(directory, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern_lower))
        image_files.extend(glob.glob(pattern_upper))

    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    return image_files

def save_generation_log(results_save_dir, args, generation_stats):
    """Save generation log to JSON file"""

    log_data = {
        'timestamp': datetime.now().isoformat(),
        'sampler_type': getattr(args, 'sampler_type', 'unknown'),
        'sampler_description': get_sampler_description(getattr(args, 'sampler_type', 'unknown')),
        'parameters': {
            'test_num': getattr(args, 'test_num', 0),
            'grid_test_num': getattr(args, 'grid_test_num', 0),
            'grid_test_index': getattr(args, 'grid_test_index', 0),
            'grid_samplers': getattr(args, 'grid_samplers', []),
            'grid_title': getattr(args, 'grid_title', ''),
            'start_index': getattr(args, 'start_index', 0),
            'batch_size': getattr(args, 'batch_size', 0),
            'num_inference_steps': getattr(args, 'num_inference_steps', 0),
            'guidance_scale': getattr(args, 'guidance', 0),
            'seed': getattr(args, 'seed', 0),
            'lamb': getattr(args, 'lamb', 0),
            'kappa': getattr(args, 'kappa', 0),
            'dtype': getattr(args, 'dtype', 'fp32'),
            'device': getattr(args, 'device', 'cpu')
        },
        'generation_stats': generation_stats
    }

    log_filename = f"generation_log_{getattr(args, 'sampler_type', 'unknown')}.json"
    log_path = os.path.join(results_save_dir, log_filename)

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"  ✓ Generation log saved to: {log_path}")

def count_generated_images(results_save_dir):
    """Count the number of generated images in the save directory"""
    try:
        image_files = find_image_files(results_save_dir)
        return len(image_files)
    except Exception as e:
        print(f"Could not count images in {results_save_dir}: {e}")
        return 0

def generate_experiment_summary(results_save_dir, args, experiment_results, start_time, end_time):
    """Generate comprehensive experiment summary report"""

    summary_file = os.path.join(results_save_dir, "experiment_summary.txt")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    successful_experiments = [r for r in experiment_results if r['success']]
    failed_experiments = [r for r in experiment_results if not r['success']]

    total_images = sum(r.get('image_count', 0) for r in successful_experiments)
    total_duration = sum(r.get('duration', 0) for r in experiment_results)

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(args.grid_title + " 实验总结报告\n")
        f.write(args.grid_title + "\n")
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
        f.write(f"- Guidance Scale: {args.guidance}\n")
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
        failed_log_file = os.path.join(results_save_dir, "failed_experiments.log")
        with open(failed_log_file, 'w', encoding='utf-8') as f:
            for result in failed_experiments:
                f.write(f"{result.get('sampler_type', 'unknown')},{result.get('num_inference_steps', 'unknown')},{result.get('error', 'Unknown error')}\n")
        print(f"⚠️  有 {len(failed_experiments)} 个实验失败，详情请查看: {failed_log_file}")
    else:
        print("🎉 所有实验都成功完成!")

def generate_comparison_grid(results_save_dir, sampler_types, num_inference_steps=20, grid_test_num=6, grid_test_index=[], grid_title="COCO Generation Comparison", coco_prompts_file=None):
    """
    生成类似截图的对比图组，方法作为行，相同图片的不同方法放在同一行

    Args:
        results_save_dir: 保存目录
        sampler_types: 采样器类型列表
        num_inference_steps: 推理步数
        grid_test_num: 测试图片数量（列数）
        grid_test_index: test pic grid index
        grid_title: 对比图组标题
        coco_prompts_file: COCO prompts文件名（可选，用于SD模型显示prompt文本）
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
    print(f"行数（方法数）: {len(sampler_types)}, 列数（图片数）: {grid_test_num}")
    print(f"测试图片索引: {grid_test_index}")

    # 确定 GeoDiff 行的索引（如果存在）
    lml_index = None
    if 'dpm_hcg' in sampler_types:
        # 优先使用，如果没有则使用dpm_lm
        lml_index = sampler_types.index('dpm_hcg') if 'dpm_hcg' in sampler_types else sampler_types.index('dpm_lm')

    # 创建图像网格：行 = 方法，列 = 测试图片
    # 所有图片大小一致（所有行的height_ratios都是1.0）
    # 通过稍微增加GeoDiff行之前一行的height_ratio来增加间距
    total_rows = len(sampler_types)
    height_ratios = [1.0] * total_rows

    # 如果存在GeoDiff行且不是第一行，稍微增加上一行的height_ratio来增加间距
    if lml_index is not None and lml_index > 0:
        # 增加上一行的height_ratio，这样可以增加GeoDiff行与上一行之间的间距
        # 使用1.15来稍微增加间距，不会太明显影响图片大小
        height_ratios[lml_index - 1] = 1.0

    fig = plt.figure(figsize=(grid_test_num * 2.5, total_rows * 2.5))
    gs = GridSpec(total_rows, grid_test_num, figure=fig,
                  hspace=0.1, wspace=0.05,
                  left=0.05, right=0.95, top=0.95, bottom=0.05,
                  height_ratios=height_ratios)

    # 方法名称映射
    method_names = {
        'ddim': 'DDIM',
        'ddim_lm': 'DDIM-LM',
        'pndm': 'PNDM',
        'dpm': 'DPM',
        'dpm++': 'DPM++',
        'unipc': 'UniPC',
        'dpm_lm': 'LML',
        'dpm_hcg': 'GeoDiff\n(Ours)'
    }

    # 为每个采样器生成图像
    all_images = {}
    max_images = 0

    for sampler_type in sampler_types:
        print(f"  生成 {sampler_type} 图像...")

        # 设置路径
        sampler_dir = os.path.join(results_save_dir, "steps" + '_' + str(num_inference_steps), sampler_type)

        # 查找该采样器生成的图像（支持 PNG 和 JPG）
        image_files = find_image_files(sampler_dir)
        if not image_files:
            print(f"  ⚠️  未找到 {sampler_type} 的图像文件")
            all_images[sampler_type] = []
            continue

        # 按文件名排序
        image_files.sort()
        max_images = max(max_images, len(image_files))
        all_images[sampler_type] = image_files

    # 确定实际要显示的图片数量和索引
    if grid_test_index and len(grid_test_index) > 0:
        # 使用指定的索引，但确保不超过grid_test_num
        actual_test_num = min(len(grid_test_index), grid_test_num)
        pic_indices = grid_test_index[:actual_test_num]  # 只取前actual_test_num个索引
    else:
        # 使用前 grid_test_num 个图片
        actual_test_num = min(grid_test_num, max_images)
        pic_indices = list(range(actual_test_num))

    # 检测是否是SD模型，并加载COCO prompts（如果需要）
    is_sd_model = False
    coco_prompts_dict = {}
    if any(keyword in results_save_dir.lower() for keyword in ['stable-diffusion', 'coco', 'sd']):
        is_sd_model = True
        # 如果提供了coco_prompts_file参数，尝试加载
        if coco_prompts_file:
            try:
                coco_prompts_path = os.path.join(project_dir, "evaluations", "coco_prompts", coco_prompts_file)
                if os.path.exists(coco_prompts_path):
                    with open(coco_prompts_path, 'r') as f:
                        coco_prompts_dict = json.load(f)
                    print(f"  ✓ 加载COCO prompts: {coco_prompts_file}")
                else:
                    print(f"  ⚠️  COCO prompts文件不存在: {coco_prompts_path}")
            except Exception as e:
                print(f"  ⚠️  无法加载COCO prompts: {e}")

    # 如果是SD模型，调整top位置为第一行的文本留出空间
    top_margin = 0.95
    if is_sd_model:
        top_margin = 1.00  # 为文本留出更多空间

    fig = plt.figure(figsize=(grid_test_num * 2.5, total_rows * 2.5))
    gs = GridSpec(total_rows, grid_test_num, figure=fig,
                  hspace=0.1, wspace=0.05,
                  left=0.05, right=0.95, top=top_margin, bottom=0.05,
                  height_ratios=height_ratios)

    # 绘制图像网格：行 = 方法，列 = 测试图片
    for row, sampler_type in enumerate(sampler_types):
        for col in range(actual_test_num):
            ax = fig.add_subplot(gs[row, col])

            # 获取图片索引
            pic_idx = pic_indices[col]

            if sampler_type in all_images and pic_idx < len(all_images[sampler_type]):
                # 加载并显示图像
                img_path = all_images[sampler_type][pic_idx]
                try:
                    img = Image.open(img_path)
                    ax.imshow(img)

                    # 如果是第一行且是SD模型，在图片上方添加prompt文本
                    if row == 0 and is_sd_model and coco_prompts_dict:
                        # 从文件名中提取key
                        filename = os.path.basename(img_path)
                        parts = filename.split('_')
                        if len(parts) >= 2:
                            key = parts[1]  # key在第一个下划线和第二个下划线之间
                            prompt_text = coco_prompts_dict.get(key, '')
                            if prompt_text:
                                # 在图片上方添加文本
                                bbox = ax.get_position()
                                text_x = bbox.x0 + bbox.width / 2  # 图片中心
                                text_y = bbox.y0 + bbox.height + 0.01  # 图片上方
                                # 限制文本长度，如果太长则截断并换行
                                max_length = 22
                                if len(prompt_text) > max_length:
                                    # 使用textwrap来换行
                                    wrapped_text = '\n'.join(textwrap.wrap(prompt_text, width=max_length))
                                    prompt_text = wrapped_text
                                fig.text(text_x, text_y, prompt_text,
                                        ha='center', va='bottom',
                                        fontsize=14, color='black',
                                        zorder=200)
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

            # 添加行标题（方法名称，只在第一列）
            if col == 0:
                method_name = method_names.get(sampler_type, sampler_type)
                # 字体大小加倍：从 12 改为 24
                # 使用 figure 坐标系来定位文字，确保在可见区域内
                bbox = ax.get_position()
                # 文字位置：在 subplot 左侧外部，使用 figure 坐标系
                text_x = bbox.x0 - 0.01  # 在 subplot 左侧稍微偏左
                text_y = bbox.y0 + bbox.height / 2  # subplot 的垂直中心

                # 如果是 GeoDiff 行，文字使用深色（因为背景只在图像区域）
                text_color = 'black'

                # 在 figure 上添加文字
                fig.text(text_x, text_y, method_name,
                        ha='right', va='center',
                        fontsize=24, fontweight='bold',
                        color=text_color, zorder=200)


    # 设置整体标题
    # fig.suptitle(f'{grid_title} (Steps: {num_inference_steps})',
    #              fontsize=16, fontweight='bold', y=0.98)

    # 保存图像
    # 获取 results_save_dir 的最后一级目录名
    last_dir_name = os.path.basename(os.path.normpath(results_save_dir))

    # 保存PNG格式（降低DPI以减小文件大小）
    output_path_png = os.path.join(results_save_dir, f'comparison_grid_{last_dir_name}_steps_{num_inference_steps}.png')
    # 使用150 DPI而不是300，可以显著减小文件大小，同时保持足够的清晰度
    # 如果需要更小的文件，可以进一步降低DPI（如100或120）
    plt.savefig(output_path_png, dpi=150, bbox_inches='tight', facecolor='white', format='png')

    # 保存PDF格式（矢量格式，适合打印和缩放）
    output_path_pdf = os.path.join(results_save_dir, f'comparison_grid_{last_dir_name}_steps_{num_inference_steps}.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', format='pdf')

    plt.close()

    print(f"✅ 对比图组已保存到:")
    print(f"   PNG: {output_path_png}")
    print(f"   PDF: {output_path_pdf}")
    return output_path_png

def generate_comparison_grid_from_existing(results_save_dir, args):
    """
    从已存在的图像文件生成对比图组

    Args:
        results_save_dir: 保存目录
        args.num_inference_steps: 推理步数
        args.grid_test_num: 测试数量（行数）
        args.grid_samplers: 采样器类型列表
        args.grid_test_index: 测试图片索引
        args.grid_title: 对比图组标题
        args.coco_prompts_file: COCO prompts文件名（可选，用于SD模型显示prompt文本）
    """
    # 定义采样器类型
    if args.grid_samplers is None:
        sampler_types = ['ddim', 'pndm', 'dpm', 'dpm++', 'unipc', 'dpm_lm', 'dpm_hcg']
    else:
        sampler_types = args.grid_samplers

    # 检查哪些采样器有图像
    available_samplers = []
    for sampler in sampler_types:
        sampler_dir = os.path.join(results_save_dir, "steps" + '_' + str(args.num_inference_steps), sampler)
        if os.path.exists(sampler_dir) and find_image_files(sampler_dir):
            available_samplers.append(sampler)

    if not available_samplers:
        print("❌ 未找到任何采样器的图像文件")
        return None

    print(f"找到 {len(available_samplers)} 个采样器的图像: {available_samplers}")

    # 获取coco_prompts_file参数（如果存在）
    coco_prompts_file = getattr(args, 'display_prompts_file', None)

    # 生成对比图组
    return generate_comparison_grid(results_save_dir, available_samplers, args.num_inference_steps, args.grid_test_num, args.grid_test_index, args.grid_title, coco_prompts_file)

# Logger functions are now imported from utils.logger
# All logger-related code has been moved to utils/logger.py
# The following functions are available via import:
# - ProjectLogger, get_logger, setup_logging
# - log_experiment_start, log_experiment_end, log_progress
# - log_error, log_success, log_warning
# - info, warning, error, debug, success, progress

# For backward compatibility: expose logger as module attribute
_logger_instance = None

def __getattr__(name):
    """Allow access to logger attribute via project.logger for backward compatibility"""
    if name == 'logger':
        # Return the logger instance's logger attribute
        logger_obj = get_logger()
        global _logger_instance
        _logger_instance = logger_obj  # Cache it
        return logger_obj.logger
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
