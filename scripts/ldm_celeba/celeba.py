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
from diffusers import LDMPipeline, DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler
from scheduler.scheduling_dpmsolver_multistep_hcg import DPMSolverMultistepHCGScheduler

import project as project

celeba_model_path = "/home/liying/Documents/ldm-celebahq-256/"
# celeba_model_path = "/research/cbim/vast/cj574/data/diffusion/celeba_hq_256/ldm-celebahq-256"

class CelebAPerformanceAnalyzer:
    """CelebA性能数据分析器"""

    def __init__(self, output_dir="output/celeba"):
        self.output_dir = output_dir
        self.results = {}
        self.performance_data = []

    def find_json_files(self):
        """查找所有JSON日志文件"""
        json_files = []

        # 查找所有steps_*目录下的JSON文件
        steps_pattern = os.path.join(self.output_dir, "steps_*", "*", "generation_log_*.json")
        json_files.extend(glob.glob(steps_pattern))

        # 查找根目录下的JSON文件
        root_pattern = os.path.join(self.output_dir, "*", "generation_log_*.json")
        json_files.extend(glob.glob(root_pattern))

        return sorted(list(set(json_files)))

    def extract_step_from_path(self, file_path):
        """从文件路径中提取步数"""
        path_parts = file_path.split(os.sep)
        for part in path_parts:
            if part.startswith("steps_"):
                return int(part.split("_")[1])
        return None

    def extract_method_from_path(self, file_path):
        """从文件路径中提取方法名"""
        path_parts = file_path.split(os.sep)
        for part in path_parts:
            if part in ["ddim", "pndm", "dpm", "dpm++", "dpm_lm", "unipc", "hessian_free"]:
                return part
        return None

    def parse_json_file(self, file_path):
        """解析JSON文件并提取性能数据"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # 提取基本信息
            step = self.extract_step_from_path(file_path)
            method = self.extract_method_from_path(file_path)

            if step is None or method is None:
                project.warning(f"Could not extract step or method from {file_path}")
                return None

            # 提取性能统计
            gen_stats = data.get('generation_stats', {})

            performance_info = {
                'file_path': file_path,
                'timestamp': data.get('timestamp', ''),
                'sampler_type': data.get('sampler_type', ''),
                'step': step,
                'method': method,
                'total_time': gen_stats.get('total_time', 0),
                'avg_time_per_batch': gen_stats.get('avg_time_per_batch', 0),
                'avg_time_per_image': gen_stats.get('avg_time_per_image', 0),
                'total_images': gen_stats.get('total_images', 0),
                'images_per_second': gen_stats.get('images_per_second', 0),
                'batch_size': data.get('parameters', {}).get('batch_size', 0),
                'num_inference_steps': data.get('parameters', {}).get('num_inference_steps', 0)
            }

            # 提取每张图片的生成时间
            generation_times = gen_stats.get('generation_times', [])
            if generation_times:
                performance_info.update({
                    'min_time_per_image': min(generation_times),
                    'max_time_per_image': max(generation_times),
                    'std_time_per_image': np.std(generation_times),
                    'median_time_per_image': np.median(generation_times),
                    'p95_time_per_image': np.percentile(generation_times, 95),
                    'p99_time_per_image': np.percentile(generation_times, 99)
                })

            return performance_info

        except Exception as e:
            project.error(f"Error parsing {file_path}: {e}")
            return None

    def analyze_performance(self):
        """分析所有性能数据"""
        project.info("🔍 查找JSON文件...")
        json_files = self.find_json_files()
        project.info(f"找到 {len(json_files)} 个JSON文件")

        project.info("\n📊 解析性能数据...")
        for file_path in json_files:
            performance_info = self.parse_json_file(file_path)
            if performance_info:
                self.performance_data.append(performance_info)

        project.info(f"成功解析 {len(self.performance_data)} 个文件")

        # 创建DataFrame
        self.df = pd.DataFrame(self.performance_data)

        if self.df.empty:
            project.warning("没有找到有效的性能数据")
            return

        # 按步数和方法分组
        self.results = self.df.groupby(['step', 'method']).agg({
            'avg_time_per_image': ['mean', 'std', 'min', 'max'],
            'images_per_second': ['mean', 'std', 'min', 'max'],
            'total_images': 'sum',
            'total_time': 'sum'
        }).round(4)

        project.success("性能数据分析完成")

    def generate_summary_table(self):
        """生成性能摘要表格"""
        if self.df.empty:
            return

        project.info("\n📋 性能摘要表格:")
        project.info("=" * 80)

        # 按步数分组显示
        for step in sorted(self.df['step'].unique()):
            project.info(f"\n步数 {step}:")
            project.info("-" * 40)
            step_data = self.df[self.df['step'] == step]

            for method in sorted(step_data['method'].unique()):
                method_data = step_data[step_data['method'] == method]
                avg_time = method_data['avg_time_per_image'].mean()
                avg_ips = method_data['images_per_second'].mean()
                total_images = method_data['total_images'].sum()

                project.info(f"  {method:12}: {avg_time:.4f}s/img, {avg_ips:.2f} img/s, {total_images} images")

    def generate_detailed_report(self, save_dir="output/celeba"):
        """生成详细报告"""
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存原始数据
        csv_path = os.path.join(save_dir, f'celeba_performance_data_{timestamp}.csv')
        self.df.to_csv(csv_path, index=False)
        project.info(f"📄 原始数据已保存到: {csv_path}")

        # 生成性能对比表格
        if not self.df.empty:
            # 按步数和方法创建对比表
            pivot_time = self.df.pivot_table(
                values='avg_time_per_image',
                index='method',
                columns='step',
                aggfunc='mean'
            ).round(4)

            pivot_ips = self.df.pivot_table(
                values='images_per_second',
                index='method',
                columns='step',
                aggfunc='mean'
            ).round(2)

            # 保存对比表
            time_csv = os.path.join(save_dir, f'celeba_time_comparison_{timestamp}.csv')
            ips_csv = os.path.join(save_dir, f'celeba_ips_comparison_{timestamp}.csv')

            pivot_time.to_csv(time_csv)
            pivot_ips.to_csv(ips_csv)

            project.info(f"📊 时间对比表已保存到: {time_csv}")
            project.info(f"📊 速度对比表已保存到: {ips_csv}")

            # 生成LaTeX表格
            self.generate_latex_table(save_dir, timestamp)

    def generate_latex_table(self, save_dir, timestamp):
        """生成LaTeX格式的性能对比表格"""
        if self.df.empty:
            return

        # 按步数和方法创建时间对比表
        pivot_time = self.df.pivot_table(
            values='avg_time_per_image',
            index='method',
            columns='step',
            aggfunc='mean'
        ).round(4)

        # 按步数和方法创建速度对比表
        pivot_ips = self.df.pivot_table(
            values='images_per_second',
            index='method',
            columns='step',
            aggfunc='mean'
        ).round(2)

        # 生成时间对比LaTeX表格
        time_latex = self.create_latex_table(pivot_time, "Average Time per Image (seconds)", "celeba_time_comparison")
        time_latex_path = os.path.join(save_dir, f'celeba_time_latex_{timestamp}.tex')
        with open(time_latex_path, 'w') as f:
            f.write(time_latex)

        # 生成速度对比LaTeX表格
        ips_latex = self.create_latex_table(pivot_ips, "Images per Second", "celeba_ips_comparison")
        ips_latex_path = os.path.join(save_dir, f'celeba_ips_latex_{timestamp}.tex')
        with open(ips_latex_path, 'w') as f:
            f.write(ips_latex)

        project.info(f"📝 LaTeX时间对比表已保存到: {time_latex_path}")
        project.info(f"📝 LaTeX速度对比表已保存到: {ips_latex_path}")

    def create_latex_table(self, pivot_df, title, label):
        """创建LaTeX表格"""
        steps = sorted(pivot_df.columns)
        methods = sorted(pivot_df.index)

        latex = f"""\\begin{{table*}}[t]
    \\centering
    \\footnotesize
    \\setlength{{\\tabcolsep}}{{2pt}}
    \\renewcommand{{\\arraystretch}}{{0.9}}
    \\begin{{tabular}}{{l|{'c' * len(steps)}}}
    \\toprule
    \\multirow{{2}}{{*}}{{Methods}} & \\multicolumn{{{len(steps)}}}{{c}}{{{title}}} \\\\
    \\cmidrule(lr){{2-{len(steps)+1}}}
    """

        # 添加列标题
        for step in steps:
            latex += f"    & {step} steps "
        latex += " \\\\\n    \\midrule\n"

        # 添加数据行
        for method in methods:
            method_escaped = method.replace("_", "\\_")
            latex += f"    {method_escaped}"
            for step in steps:
                value = pivot_df.loc[method, step]
                if pd.isna(value):
                    latex += " & N/A "
                else:
                    latex += f" & {value:.4f} "
            latex += " \\\\\n"

        latex += f"""    \\bottomrule
    \\end{{tabular}}
    \\caption{{{title} comparison across different sampling methods and inference steps on CelebA-HQ generation.}}
    \\label{{tab_{label}}}
\\end{{table*}}"""

        return latex

    def print_statistics(self):
        """打印统计信息"""
        if self.df.empty:
            project.warning("没有数据可显示")
            return

        project.info("\n📈 统计信息:")
        project.info("=" * 50)

        # 总体统计
        project.info(f"总实验数: {len(self.df)}")
        project.info(f"步数范围: {self.df['step'].min()} - {self.df['step'].max()}")
        project.info(f"方法数: {len(self.df['method'].unique())}")
        project.info(f"总生成图片数: {self.df['total_images'].sum()}")
        project.info(f"总耗时: {self.df['total_time'].sum():.2f} 秒")

        # 最快和最慢的方法
        fastest_method = self.df.loc[self.df['images_per_second'].idxmax()]
        slowest_method = self.df.loc[self.df['images_per_second'].idxmin()]

        project.info(f"\n🚀 最快方法: {fastest_method['method']} (步数{fastest_method['step']}) - {fastest_method['images_per_second']:.2f} img/s")
        project.info(f"🐌 最慢方法: {slowest_method['method']} (步数{slowest_method['step']}) - {slowest_method['images_per_second']:.2f} img/s")

def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="CelebA-HQ sampling script with enhanced features")

    # Basic parameters
    parser.add_argument('--test_num', type=int, default=20)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=20, choices=[5, 10, 20, 30, 40, 50, 80, 100])

    # Sampler selection
    parser.add_argument('--sampler_type', type=str, default='hessian_free',
                        choices=['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'hessian_free'])
    parser.add_argument('--use_generator', action='store_true', default=True)

    # Output configuration
    parser.add_argument('--save_dir', type=str, default='celeba')
    parser.add_argument('--model_path', type=str, default=celeba_model_path)
    parser.add_argument('--model_type', type=str, default="ldm")

    # LML parameters
    parser.add_argument('--lamb', type=float, default=0.004)
    parser.add_argument('--kappa', type=float, default=1.0e-8)

    # Technical parameters
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp64', 'fp16', 'bf16'])
    parser.add_argument('--device', type=str, default='cuda')

    # Evaluation options
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation metrics')
    parser.add_argument('--generate_grid', action='store_true', default=True, help='Generate comparison grid from existing images')
    parser.add_argument('--grid_title', type=str, default="CelebA-HQ Generation Comparison", help='Title of comparison grid')
    parser.add_argument('--grid_test_num', type=int, default=6, help='Number of images to test in grid')
    parser.add_argument('--grid_test_index', type=list, default=[0, 1, 2, 3, 4, 5], help='Index of images to test in grid')
    parser.add_argument('--grid_samplers', default=['ddim', 'pndm', 'dpm++', 'dpm', 'unipc', 'hessian_free'],
                        help='List of samplers to test in batch mode')

    # Batch processing options
    parser.add_argument('--run_batch', action='store_true', default=False, help='Run batch experiments with multiple samplers and steps')
    parser.add_argument('--run_batch_samplers', default=['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'hessian_free'], help='List of samplers to test in batch mode')
    parser.add_argument('--run_batch_steps', type=int, default=[10, 20, 50], help='List of inference steps to test in batch mode')

    # Additional options
    parser.add_argument('--save_log', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true')

    # Performance analysis options
    parser.add_argument('--analyze_performance', action='store_true', default=False, help='Run performance analysis after experiments')
    parser.add_argument('--performance_output_dir', type=str, default='output/celeba', help='Directory for performance analysis output')
    parser.add_argument('--performance_only', action='store_true', default=False, help='Only run performance analysis on existing results')

    args = parser.parse_args()

    return args

def setup_scheduler(pipe, sampler_type, lamb=0.0008, kappa=1e-8):
    """Setup the appropriate scheduler based on sampler type"""

    if sampler_type == 'pndm':
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        project.info(f"  Using PNDM scheduler")

    elif sampler_type == 'ddim':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        project.info(f"  Using DDIM scheduler")

    elif sampler_type == 'ddim_lm':
        pipe.scheduler = DDIMLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = True
        pipe.scheduler.kappa = kappa
        project.info(f"  Using DDIM with LML correction (λ={lamb}, κ={kappa})")

    elif sampler_type == 'dpm++':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.config.algorithm_type = "dpmsolver++"
        project.info(f"  Using DPM-Solver++ scheduler")

    elif sampler_type == 'dpm_lm':
        pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.config.algorithm_type = "dpmsolver"
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = True
        pipe.scheduler.kappa = kappa
        project.info(f"  Using DPM-Solver with LML correction (λ={lamb}, κ={kappa})")

    elif sampler_type == 'dpm':
        pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.config.algorithm_type = "dpmsolver"
        pipe.scheduler.lm = False
        project.info(f"  Using DPM-Solver scheduler")

    elif sampler_type == 'unipc':
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        project.info(f"  Using UniPC scheduler")

    elif sampler_type == 'hessian_free':
        pipe.scheduler = DPMSolverMultistepHCGScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.solver_order = 3
        pipe.scheduler.config.algorithm_type = "dpmsolver++"
        pipe.scheduler.lamb = lamb
        pipe.scheduler.lm = True
        pipe.scheduler.kappa = kappa
        pipe.scheduler.hessian_method = 'hessian_free'
        # 设置模型用于Hessian计算
        pipe.scheduler.set_model(pipe.unet)
        project.info(f"  Using DPM-Solver++ with Hessian-Free LML correction (λ={lamb}, κ={kappa})")

    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

def safe_array_conversion(img):
    """Safely convert PIL Image to numpy array without deprecation warnings"""
    if isinstance(img, Image.Image):
        # Convert PIL Image to numpy array without copy parameter
        return np.asarray(img)
    else:
        return img

def calculate_color_score(images):
    """Calculate ColorS metric - Colorfulness score"""
    color_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Calculate colorfulness using standard deviation of color channels
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

            # Calculate mean and standard deviation for each channel
            mean_r, std_r = np.mean(r), np.std(r)
            mean_g, std_g = np.mean(g), np.std(g)
            mean_b, std_b = np.mean(b), np.std(b)

            # Colorfulness metric (simplified version)
            colorfulness = np.sqrt(std_r**2 + std_g**2 + std_b**2)
            color_scores.append(colorfulness)

    return np.mean(color_scores) if color_scores else 0.0

def calculate_face_score(images):
    """Calculate FS metric - Face Score (simplified)"""
    face_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to grayscale for face detection
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Simple face quality metric based on edge detection and symmetry
        # This is a simplified version - in practice, you'd use a trained face quality model

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # Symmetry (left-right)
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])

        if left_half.shape == right_half.shape:
            symmetry = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
        else:
            symmetry = 0.5

        # Combined face score
        face_score = (edge_density * 0.3 + symmetry * 0.7) * 10  # Scale to match expected range
        face_scores.append(face_score)

    return np.mean(face_scores) if face_scores else 0.0

def calculate_df_iqa(images):
    """Calculate DFIQA metric - Deep Face Image Quality Assessment (simplified)"""
    df_iqa_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Calculate image quality metrics
        # Sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Contrast
        contrast = gray.std()

        # Brightness distribution
        brightness = gray.mean()

        # Combined quality score (simplified)
        quality_score = (laplacian_var / 1000.0 + contrast / 100.0 + brightness / 255.0) / 3.0
        df_iqa_scores.append(quality_score)

    return np.mean(df_iqa_scores) if df_iqa_scores else 0.0

def calculate_pic_score(images):
    """Calculate PicS metric - Picture Score (aesthetic quality)"""
    pic_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Calculate aesthetic quality metrics

            # Color harmony (simplified)
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            color_balance = 1.0 - np.std([r.mean(), g.mean(), b.mean()]) / 255.0

            # Composition (rule of thirds approximation)
            h, w = img_array.shape[:2]
            center_region = img_array[h//3:2*h//3, w//3:2*w//3]
            composition_score = center_region.std() / img_array.std() if img_array.std() > 0 else 0.5

            # Overall aesthetic score
            aesthetic_score = (color_balance * 0.4 + composition_score * 0.6) * 25  # Scale to match expected range
            pic_scores.append(aesthetic_score)

    return np.mean(pic_scores) if pic_scores else 0.0

def calculate_eat_score(images):
    """Calculate EAT metric - Enhanced Aesthetic Test (simplified)"""
    eat_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Calculate enhanced aesthetic metrics

            # Color diversity
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
            color_diversity = unique_colors / (img_array.shape[0] * img_array.shape[1]) * 1000

            # Edge complexity
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_complexity = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]) * 100

            # Combined EAT score
            eat_score = (color_diversity * 0.3 + edge_complexity * 0.7) * 0.05  # Scale to match expected range
            eat_scores.append(eat_score)

    return np.mean(eat_scores) if eat_scores else 0.0

def calculate_laion_score(images):
    """Calculate Laion metric - LAION aesthetic score (simplified)"""
    laion_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Calculate LAION-style aesthetic metrics

            # Image clarity
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            clarity = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Color saturation
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:, :, 1])

            # Overall aesthetic quality
            laion_score = (clarity / 1000.0 + saturation / 255.0) * 2.5  # Scale to match expected range
            laion_scores.append(laion_score)

    return np.mean(laion_scores) if laion_scores else 0.0

def evaluate_images(images, sampler_type):
    """Evaluate images using all metrics"""
    project.info(f"\n📊 Evaluating {len(images)} images for {sampler_type}...")

    # Calculate all metrics
    color_s = calculate_color_score(images)
    fs = calculate_face_score(images)
    df_iqa = calculate_df_iqa(images)
    pic_s = calculate_pic_score(images)
    eat = calculate_eat_score(images)
    laion = calculate_laion_score(images)

    results = {
        'sampler_type': sampler_type,
        'num_images': len(images),
        'ColorS': color_s,
        'FS': fs,
        'DFIQA': df_iqa,
        'PicS': pic_s,
        'EAT': eat,
        'Laion': laion
    }

    project.info(f"  ColorS: {color_s:.3f}")
    project.info(f"  FS: {fs:.3f}")
    project.info(f"  DFIQA: {df_iqa:.3f}")
    project.info(f"  PicS: {pic_s:.3f}")
    project.info(f"  EAT: {eat:.3f}")
    project.info(f"  Laion: {laion:.3f}")

    return results

def generate_comparison_table(results_dict):
    """Generate comparison table in the format shown in the image"""

    # Define the methods in order
    methods = ['DDIM [75]', 'PNDM [50]', 'DPM [51]', 'DPM++ [52]', 'UniPC [91]', 'HILDA(Ours)']

    # Map sampler types to method names
    sampler_to_method = {
        'ddim': 'DDIM [75]',
        'pndm': 'PNDM [50]',
        'dpm': 'DPM [51]',
        'dpm++': 'DPM++ [52]',
        'unipc': 'UniPC [91]',
        # 'dpm_lm': 'LML',
        # 'ddim_lm': 'LML',
        'hessian_free': 'HILDA(Ours)'
    }

    # Create table data
    table_data = []
    for method in methods:
        # Find corresponding results
        method_results = None
        for sampler_type, results in results_dict.items():
            if sampler_to_method.get(sampler_type) == method:
                method_results = results
                break

        if method_results:
            table_data.append({
                'Method': method,
                'ColorS': method_results['ColorS'],
                'FS': method_results['FS'],
                'DFIQA': method_results['DFIQA'],
                'PicS': method_results['PicS'],
                'EAT': method_results['EAT'],
                'Laion': method_results['Laion']
            })

    return table_data

def format_table_with_ranking(table_data):
    """Format table with best and second-best highlighting"""

    # Extract metrics for ranking
    metrics = ['ColorS', 'FS', 'DFIQA', 'PicS', 'EAT', 'Laion']

    # Find best and second-best for each metric
    rankings = {}
    for metric in metrics:
        values = [(i, row[metric]) for i, row in enumerate(table_data)]
        values.sort(key=lambda x: x[1], reverse=True)  # Higher is better

        if len(values) >= 2:
            best_idx = values[0][0]
            second_best_idx = values[1][0]
            rankings[metric] = {'best': best_idx, 'second': second_best_idx}

    # Create formatted table
    project.info("\n" + "="*120)
    project.info("Table 2. Comparison of different samplers on CelebA-HQ unconditional generation.")
    project.info("Best results are bolded and the second best results are underlined.")
    project.info("="*120)

    # Header
    header = f"{'Methods':<15} {'Colorful':<20} {'Face Quality':<25} {'Aesthetic':<30}"
    project.info(header)
    subheader = f"{'':15} {'ColorS(↑)':<10} {'FS(↑)':<10} {'DFIQA(↑)':<10} {'PicS(↑)':<10} {'EAT(↑)':<10} {'Laion(↑)':<10}"
    project.info(subheader)
    project.info("-" * 120)

    # Data rows
    for i, row in enumerate(table_data):
        method = row['Method']
        line = f"{method:<15}"

        for metric in metrics:
            value = row[metric]
            formatted_value = f"{value:.3f}"

            # Apply formatting
            if metric in rankings:
                if i == rankings[metric]['best']:
                    formatted_value = f"**{formatted_value}**"  # Bold
                elif i == rankings[metric]['second']:
                    formatted_value = f"_{formatted_value}_"    # Underline

            line += f" {formatted_value:<10}"

        project.info(line)

    project.info("="*120)

def generate_images(results_save_dir, args, pipe):
    """Generate images using the specified pipeline"""

    total_time = 0
    generation_times = []
    all_images = []

    project.info(f"\n{'='*60}")
    project.info(f"Starting image generation")
    project.info(f"{'='*60}")
    project.info(f"Sampler: {args.sampler_type} - {project.get_sampler_description(args.sampler_type)}")
    project.info(f"Batch size: {args.batch_size}")
    project.info(f"Inference steps: {args.num_inference_steps}")
    project.info(f"Total images: {args.test_num}")
    project.info(f"Save directory: {results_save_dir}")
    project.info(f"{'='*60}")

    for seed in range(args.seed, args.seed + args.test_num):
        project.info(f"\nGenerating batch {seed - args.seed + 1}/{args.test_num} (seed={seed})")
        batch_start_time = time.time()
        torch.manual_seed(seed)

        # Generate images
        # 确保hessian_free方法有正确的模型设置
        if hasattr(pipe.scheduler, 'set_model') and pipe.scheduler.model is None:
            pipe.scheduler.set_model(pipe.unet)

        with torch.no_grad():
            images = pipe(batch_size=args.batch_size, num_inference_steps=args.num_inference_steps).images

        # Save images
        for i, image in enumerate(images):
            filename = f"celeba_{args.sampler_type}_inference{args.num_inference_steps}_seed{seed}_{i}.png"
            filepath = os.path.join(results_save_dir, filename)
            image.save(filepath)
            all_images.append(image)

        batch_time = time.time() - batch_start_time
        generation_times.append(batch_time)
        total_time += batch_time

        project.info(f"  ✓ Generated {len(images)} images in {batch_time:.3f}s")
        project.info(f"  ✓ Saved to: {results_save_dir}")

    # Print summary
    avg_time = total_time / args.test_num
    avg_time_per_image = avg_time / args.batch_size

    project.info(f"\n{'='*60}")
    project.info(f"GENERATION SUMMARY")
    project.info(f"{'='*60}")
    project.info(f"Total time: {total_time:.2f}s")
    project.info(f"Average time per batch: {avg_time:.3f}s")
    project.info(f"Average time per image: {avg_time_per_image:.3f}s")
    project.info(f"Total images generated: {args.test_num * args.batch_size}")
    project.info(f"Images per second: {args.test_num * args.batch_size / total_time:.2f}")
    project.info(f"{'='*60}")

    # Evaluate images if requested
    evaluation_results = None
    if args.evaluate and all_images:
        evaluation_results = evaluate_images(all_images, args.sampler_type)

    return {
        'total_time': total_time,
        'avg_time_per_batch': avg_time,
        'avg_time_per_image': avg_time_per_image,
        'total_images': args.test_num * args.batch_size,
        'images_per_second': args.test_num * args.batch_size / total_time,
        'generation_times': generation_times,
        'evaluation_results': evaluation_results
    }

def run_single_experiment(results_save_dir, args, experiment_num, total_experiments):
    """Run a single experiment with enhanced logging"""

    project.info("")
    project.info("="*50)
    project.info(f"实验 {experiment_num}/{total_experiments}")
    project.info(f"Sampler: {args.sampler_type}")
    project.info(f"Steps: {args.num_inference_steps}")
    project.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    project.info("="*50)

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

        # Setup paths
        model_path = args.model_path
        save_dir = os.path.join(project.output_dir, args.save_dir, "steps"+'_'+str(args.num_inference_steps), args.sampler_type)
        os.makedirs(save_dir, exist_ok=True)

        project.info(f"🚀 CelebA-HQ Unified Sampling Script")
        project.info("="*60)
        project.info(f"Model: {model_path}")
        project.info(f"Device: {args.device}")
        project.info(f"Data type: {args.dtype}")
        project.info(f"Sampler: {args.sampler_type}")
        project.info(f"Output: {save_dir}")
        project.info("="*60)

        # Load pipeline
        project.info("\n📦 Loading model...")
        pipe = LDMPipeline.from_pretrained(model_path, torch_dtype=dtype, use_safetensors=False)
        pipe.unet.to(args.device)
        pipe.vqvae.to(args.device)
        project.success("Model loaded successfully")

        # Setup scheduler
        project.info(f"\n⚙️ Setting up scheduler...")
        setup_scheduler(pipe, args.sampler_type, args.lamb, args.kappa)

        # Generate images
        generation_stats = generate_images(
            results_save_dir, args, pipe
        )

        # Save generation log if requested
        if args.save_log:
            project.info(f"\n📝 Saving generation log...")
            project.save_generation_log(results_save_dir, args, generation_stats)

        # 计算实验耗时
        exp_end_time = time.time()
        exp_duration = exp_end_time - exp_start_time

        # 统计生成的图片数量
        image_count = project.count_generated_images(results_save_dir)

        project.success(f"实验完成! 耗时: {exp_duration:.1f}秒")
        project.info(f"生成图片数量: {image_count}")
        project.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            'success': True,
            'duration': exp_duration,
            'image_count': image_count,
            'generation_stats': generation_stats,
            'results_save_dir': results_save_dir
        }

    except Exception as e:
        exp_end_time = time.time()
        exp_duration = exp_end_time - exp_start_time

        project.error(f"实验失败! 耗时: {exp_duration:.1f}秒")
        project.error(f"错误信息: {str(e)}")
        project.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            'success': False,
            'duration': exp_duration,
            'error': str(e),
            'sampler_type': args.sampler_type,
            'num_inference_steps': args.num_inference_steps
        }

def run_performance_analysis(output_dir, save_dir):
    """运行性能分析"""
    project.info("\n" + "="*60)
    project.info("开始性能分析")
    project.info("="*60)

    try:
        # 创建性能分析器
        analyzer = CelebAPerformanceAnalyzer(output_dir)

        # 分析性能数据
        analyzer.analyze_performance()

        # 生成摘要表格
        analyzer.generate_summary_table()

        # 打印统计信息
        analyzer.print_statistics()

        # 生成详细报告
        analyzer.generate_detailed_report(save_dir)

        project.success("性能分析完成！")
        return True

    except Exception as e:
        project.error(f"性能分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 设置日志系统
    logger = project.setup_logging(name='celeba', level=project.logging.INFO)

    args = parse_args()
    results_save_dir = os.path.join(project.output_dir, args.save_dir + '_' + args.model_type)

    # 如果只是运行性能分析
    if args.performance_only:
        project.info("🚀 CelebA性能分析模式")
        project.info("="*50)

        performance_success = run_performance_analysis(
            args.performance_output_dir,
            results_save_dir
        )

        if performance_success:
            project.success("性能分析完成！")
        else:
            project.error("性能分析失败")

        project.info(f"日志文件: {logger.get_log_file()}")
        logger.close()
        sys.exit(0)

    # 单个实验模式（保持原有逻辑）
    SAMPLER_TYPES = [args.sampler_type]
    INFERENCE_STEPS = [args.num_inference_steps]

    # 检查是否运行批量实验
    if hasattr(args, 'run_batch') and args.run_batch:
        # 批量实验模式
        SAMPLER_TYPES = args.run_batch_samplers  #['pndm', 'ddim_lm', 'ddim', 'dpm++', 'dpm', 'dpm_lm', 'unipc', 'hessian_free']
        INFERENCE_STEPS = args.run_batch_steps  # [5, 10, 20, 30]

    total_experiments = len(SAMPLER_TYPES) * len(INFERENCE_STEPS)
    experiment_results = []
    start_time = datetime.now()
    experiment_num = 0

    project.log_experiment_start("CelebA-HQ 批量实验", {
        'samplers': SAMPLER_TYPES,
        'steps': args.num_inference_steps,
        'batch_size': args.batch_size,
        'test_num': args.test_num
    })

    print("="*50)
    print("CelebA-HQ 实验批量运行开始")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    for i in range(len(INFERENCE_STEPS)):
        for j in range(len(SAMPLER_TYPES)):
            experiment_num += 1
            num_inference_steps = INFERENCE_STEPS[i]
            sampler_type = SAMPLER_TYPES[j]

            # 更新args
            args.num_inference_steps = num_inference_steps
            args.sampler_type = sampler_type

            project.progress(experiment_num, total_experiments, f"Running {sampler_type} with {num_inference_steps} steps")

            result = run_single_experiment(results_save_dir, args, experiment_num, total_experiments)
            if result['success']:
                generation_stats = result['generation_stats']
                project.success("Generation completed successfully!")
                project.info(f"   Generated {generation_stats['total_images']} images")
                project.info(f"   Total time: {generation_stats['total_time']:.2f}s")
                project.info(f"   Average time per image: {generation_stats['avg_time_per_image']:.3f}s")

                if generation_stats.get('evaluation_results'):
                    project.info(f"\n📊 Evaluation Results:")
                    eval_results = generation_stats['evaluation_results']
                    project.info(f"   ColorS: {eval_results['ColorS']:.3f}")
                    project.info(f"   FS: {eval_results['FS']:.3f}")
                    project.info(f"   DFIQA: {eval_results['DFIQA']:.3f}")
                    project.info(f"   PicS: {eval_results['PicS']:.3f}")
                    project.info(f"   EAT: {eval_results['EAT']:.3f}")
                    project.info(f"   Laion: {eval_results['Laion']:.3f}")

            experiment_results.append(result)

        if args.generate_grid:
            # 生成对比图组模式
            print("\n🎨 生成对比图组...")
            output_path = project.generate_comparison_grid_from_existing(results_save_dir, args)
            if output_path:
                print(f"✅ 对比图组已生成: {output_path}")
            else:
                print("❌ 对比图组生成失败")

    # Generate comparison table
    if experiment_results:
        table_data = generate_comparison_table(experiment_results)
        format_table_with_ranking(table_data)

        results_file = os.path.join(results_save_dir, 'celeba_comparison_results.json')
        with open(results_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        project.info(f"\n📊 Results saved to: {results_file}")

    end_time = datetime.now()

    print("\n" + "="*50)
    print("CelebA-HQ 实验批量运行完成")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    # 生成实验总结报告
    project.info("\n生成实验总结报告...")
    project.generate_experiment_summary(results_save_dir, args, experiment_results, start_time, end_time)

    # 如果启用了性能分析，在总结中也包含性能信息
    if args.analyze_performance:
        project.info("\n在实验总结中添加性能分析结果...")
        try:
            analyzer = CelebAPerformanceAnalyzer(args.performance_output_dir)
            analyzer.analyze_performance()
            if not analyzer.df.empty:
                # 生成性能摘要并添加到总结中
                performance_summary = {
                    'total_experiments': len(analyzer.df),
                    'methods_tested': len(analyzer.df['method'].unique()),
                    'steps_tested': sorted(analyzer.df['step'].unique()),
                    'total_images': analyzer.df['total_images'].sum(),
                    'total_time': analyzer.df['total_time'].sum(),
                    'fastest_method': analyzer.df.loc[analyzer.df['images_per_second'].idxmax()]['method'],
                    'fastest_speed': analyzer.df['images_per_second'].max(),
                    'slowest_method': analyzer.df.loc[analyzer.df['images_per_second'].idxmin()]['method'],
                    'slowest_speed': analyzer.df['images_per_second'].min()
                }

                # 保存性能摘要到JSON文件
                performance_summary_path = os.path.join(results_save_dir, 'performance_summary.json')
                with open(performance_summary_path, 'w') as f:
                    json.dump(performance_summary, f, indent=2)
                project.info(f"📊 性能摘要已保存到: {performance_summary_path}")
        except Exception as e:
            project.warning(f"生成性能摘要时出错: {e}")

    project.log_experiment_end("CelebA-HQ 批量实验",
                                duration=(end_time - start_time).total_seconds(),
                                results={'total_experiments': total_experiments, 'successful': len([r for r in experiment_results if r['success']])})

    # 运行性能分析（如果启用）
    if args.analyze_performance:
        project.info("\n" + "="*60)
        project.info("开始性能分析")
        project.info("="*60)

        performance_success = run_performance_analysis(
            args.performance_output_dir,
            results_save_dir
        )

        if performance_success:
            project.success("性能分析完成！")
        else:
            project.warning("性能分析失败，但实验已完成")

    project.info(f"\n实验完成! 结果保存在: {results_save_dir}")
    project.info(f"日志文件: {logger.get_log_file()}")

    # 关闭日志器
    logger.close()
