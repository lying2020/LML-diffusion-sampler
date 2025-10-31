import os
import logging
import sys, os, json, glob
import json
from datetime import datetime
from pathlib import Path

# 项目路径配置
project_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_dir, 'model')
data_dir = os.path.join(project_dir, 'data')
output_dir = os.path.join(project_dir, 'output')
output_test_dir = os.path.join(output_dir, 'test')



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
        'hcg': 'DPM-Solver with Hessian-Free HVP correction using CG'
    }
    return descriptions.get(sampler_type, 'Unknown sampler type')

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

    log_filename = f"generation_log_{getattr(args, 'sampler_type', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path = os.path.join(results_save_dir, log_filename)

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"  ✓ Generation log saved to: {log_path}")

def count_generated_images(results_save_dir):
    """Count the number of generated images in the save directory"""
    try:
        png_files = glob.glob(os.path.join(results_save_dir, "*.png"))
        return len(png_files)
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

def generate_comparison_grid(results_save_dir, sampler_types, num_inference_steps=20, grid_test_num=6, grid_test_index=[], grid_title="COCO Generation Comparison"):
    """
    生成类似截图的对比图组，6行多列展示不同采样方法的结果

    Args:
        results_save_dir: 保存目录
        sampler_types: 采样器类型列表
        num_inference_steps: 推理步数
        grid_test_num: 测试数量（行数）
        grid_test_index: test pic grid index
        grid_title: 对比图组标题
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
    print(f"行数: {grid_test_num}, 列数: {len(sampler_types)}")
    print(f"测试图片索引: {grid_test_index}")

    # 创建图像网格
    fig = plt.figure(figsize=(len(sampler_types) * 2.5, grid_test_num * 2.5))
    gs = GridSpec(grid_test_num, len(sampler_types), figure=fig,
                  hspace=0.1, wspace=0.05,
                  left=0.05, right=0.95, top=0.95, bottom=0.05)

    # 方法名称映射
    method_names = {
        'ddim': 'DDIM',
        'ddim_lm': 'DDIM-LM',
        'pndm': 'PNDM',
        'dpm': 'DPM-Solver',
        'dpm++': 'DPM-Solver++',
        'unipc': 'UniPC',
        'dpm_lm': 'LML',
        'hcg': 'HILDA (Ours)'
    }

    # 为每个采样器生成图像
    all_images = {}

    for col, sampler_type in enumerate(sampler_types):
        print(f"  生成 {sampler_type} 图像...")

        # 设置路径
        sampler_dir = os.path.join(results_save_dir, "steps" + '_' + str(num_inference_steps), sampler_type)

        # 查找该采样器生成的图像
        image_files = glob.glob(os.path.join(sampler_dir, "*.png"))
        if not image_files:
            print(f"  ⚠️  未找到 {sampler_type} 的图像文件")
            continue

        grid_test_num = min(grid_test_num, len(image_files))
        # 按文件名排序，取前test_num个
        image_files.sort()
        selected_images = image_files[:grid_test_num]
        if grid_test_index and max(grid_test_index) < grid_test_num:
            selected_images = [image_files[i] for i in grid_test_index]
        all_images[sampler_type] = selected_images

    # 绘制图像网格
    for row in range(grid_test_num):
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

            # # 添加行标签（只在第一列）
            # if col == 0:
            #     ax.text(-0.15, 0.5, f'Row {row+1}', ha='center', va='center',
            #            transform=ax.transAxes, fontsize=10, rotation=90)

    # 添加分隔线（在LML或HILDA列后）
    if 'dpm_lm' in sampler_types or 'hcg' in sampler_types:
        # 优先使用，如果没有则使用dpm_lm
        lml_index = sampler_types.index('hcg') if 'hcg' in sampler_types else sampler_types.index('dpm_lm')
        if lml_index < len(sampler_types) - 1:
            # 在LML/HILDA列后添加垂直分隔线
            for row in range(grid_test_num):
                ax = fig.add_subplot(gs[row, lml_index])
                # 添加右侧边框
                ax.add_patch(patches.Rectangle((0.95, 0), 0.05, 1,
                                             transform=ax.transAxes,
                                             facecolor='black', alpha=0.3))

    # 设置整体标题
    fig.suptitle(f'COCO Generation Comparison (Steps: {num_inference_steps})',
                 fontsize=16, fontweight='bold', y=0.98)

    # 保存图像
    output_path = os.path.join(results_save_dir, f'comparison_grid_steps_{num_inference_steps}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ 对比图组已保存到: {output_path}")
    return output_path

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
    """
    # 定义采样器类型
    if args.grid_samplers is None:
        sampler_types = ['ddim', 'pndm', 'dpm', 'dpm++', 'unipc', 'dpm_lm', 'hcg']
    else:
        sampler_types = args.grid_samplers

    # 检查哪些采样器有图像
    available_samplers = []
    for sampler in sampler_types:
        sampler_dir = os.path.join(results_save_dir, "steps" + '_' + str(args.num_inference_steps), sampler)
        if os.path.exists(sampler_dir) and glob.glob(os.path.join(sampler_dir, "*.png")):
            available_samplers.append(sampler)

    if not available_samplers:
        print("❌ 未找到任何采样器的图像文件")
        return None

    print(f"找到 {len(available_samplers)} 个采样器的图像: {available_samplers}")

    # 生成对比图组
    return generate_comparison_grid(results_save_dir, available_samplers, args.num_inference_steps, args.grid_test_num, args.grid_test_index, args.grid_title)

class ProjectLogger:
    """项目统一的日志管理器"""

    def __init__(self, name=None, log_dir=None, level=logging.INFO):
        self.name = name or 'project'
        self.log_dir = log_dir or os.path.join(output_dir, 'logs')
        self.level = level

        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)

        # 设置日志文件名（包含时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f'{self.name}_{timestamp}.log')

        # 初始化日志器
        self.logger = self._setup_logger()

        # 保存原始print函数
        self._original_print = print

        # 重定向print到日志
        self._redirect_print()

    def _setup_logger(self):
        """设置日志器配置"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)

        # 清除已有的处理器
        logger.handlers.clear()

        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(self.level)

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _redirect_print(self):
        """重定向print函数到日志"""
        def log_print(*args, **kwargs):
            # 将print的内容转换为字符串
            message = ' '.join(str(arg) for arg in args)
            self.logger.info(message)
            # 同时输出到控制台
            self._original_print(*args, **kwargs)

        # 替换全局print函数
        import builtins
        builtins.print = log_print

    def info(self, message):
        """记录信息日志"""
        self.logger.info(message)

    def warning(self, message):
        """记录警告日志"""
        self.logger.warning(message)

    def error(self, message):
        """记录错误日志"""
        self.logger.error(message)

    def debug(self, message):
        """记录调试日志"""
        self.logger.debug(message)

    def critical(self, message):
        """记录严重错误日志"""
        self.logger.critical(message)

    def get_log_file(self):
        """获取日志文件路径"""
        return self.log_file

    def close(self):
        """关闭日志器"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

# 全局日志器实例
_global_logger = None

def get_logger(name=None, log_dir=None, level=logging.INFO):
    """获取全局日志器实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = ProjectLogger(name, log_dir, level)
    return _global_logger

def setup_logging(name=None, log_dir=None, level=logging.INFO):
    """设置项目日志系统"""
    global _global_logger
    if _global_logger is not None:
        _global_logger.close()
    _global_logger = ProjectLogger(name, log_dir, level)
    return _global_logger

def log_experiment_start(experiment_name, parameters=None):
    """记录实验开始"""
    logger = get_logger()
    logger.info("="*60)
    logger.info(f"🚀 实验开始: {experiment_name}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if parameters:
        logger.info("实验参数:")
        for key, value in parameters.items():
            logger.info(f"  {key}: {value}")
    logger.info("="*60)

def log_experiment_end(experiment_name, duration=None, results=None):
    """记录实验结束"""
    logger = get_logger()
    logger.info("="*60)
    logger.info(f"✅ 实验完成: {experiment_name}")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if duration:
        logger.info(f"总耗时: {duration:.2f}秒")
    if results:
        logger.info("实验结果:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")
    logger.info("="*60)

def log_progress(current, total, message=""):
    """记录进度信息"""
    logger = get_logger()
    percentage = (current / total) * 100 if total > 0 else 0
    logger.info(f"进度: {current}/{total} ({percentage:.1f}%) - {message}")

def log_error(error, context=""):
    """记录错误信息"""
    logger = get_logger()
    logger.error(f"❌ 错误 {context}: {str(error)}")

def log_success(message):
    """记录成功信息"""
    logger = get_logger()
    logger.info(f"✅ {message}")

def log_warning(message):
    """记录警告信息"""
    logger = get_logger()
    logger.warning(f"⚠️ {message}")

# 便捷函数
def info(message):
    """记录信息"""
    get_logger().info(message)

def warning(message):
    """记录警告"""
    get_logger().warning(message)

def error(message):
    """记录错误"""
    get_logger().error(message)

def debug(message):
    """记录调试信息"""
    get_logger().debug(message)

def success(message):
    """记录成功信息"""
    log_success(message)

def progress(current, total, message=""):
    """记录进度"""
    log_progress(current, total, message)
