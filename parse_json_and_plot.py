
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
import project as project



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

if __name__ == "__main__":

    # 设置日志系统
    logger = project.setup_logging(name='celeba', level=project.logging.INFO)

    save_dir = "celeba"
    model_type = "ldm"
    results_save_dir = os.path.join(project.output_dir, save_dir + '_' + model_type)

    output_dir = results_save_dir
    performance_save_dir = os.path.join(project.output_dir, 'performance')
    run_performance_analysis(output_dir, save_dir)


    # 如果只是运行性能分析
    project.info("🚀 CelebA性能分析模式")
    project.info("="*50)

    performance_success = run_performance_analysis(
        performance_save_dir,
        results_save_dir
    )

    if performance_success:
        project.success("性能分析完成！")
    else:
        project.warning("性能分析失败，但实验已完成")

        project.info(f"日志文件: {logger.get_log_file()}")
        logger.close()
        sys.exit(0)

    # 如果启用了性能分析，在总结中也包含性能信息

    project.info("\n在实验总结中添加性能分析结果...")

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
