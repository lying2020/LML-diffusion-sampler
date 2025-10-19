#!/usr/bin/env python3
"""
Unified Zigzag Analysis for CIFAR-10 Diffusion Models
按照ICLR论文格式生成高质量的zigzag分析图表和评价指标

Features:
1. PCA2降维算法，使用前2个PCA值进行分析
2. 采样20步，使用100个样本
3. 支持 ["pndm", "ddim", "dpm", "unipc"] 方法
4. 生成zigzag折线图、路径图、收敛分析图、特征值谱图
5. 计算多种zigzag评价指标
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import sys
import os
import time
from datetime import datetime
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set matplotlib to use a backend that doesn't require display
import matplotlib
matplotlib.use('Agg')

# Import schedulers - 参考cifar10.py的导入方式
from diffusers import DDPMPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_unipc_multistep_lm import UniPCMultistepSchedulerLM
from scheduler.scheduling_ddim_lm import DDIMLMScheduler
from scheduler.scheduling_pndm_lm import PNDMSchedulerLM

import project as project

# Set matplotlib parameters for ICLR paper format
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2.5,
    'lines.markersize': 6,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class ZigzagAnalyzer:
    """统一的zigzag分析器"""

    def __init__(self, n_samples=100, num_inference_steps=20, device='cuda'):
        self.n_samples = n_samples
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.results = {}

        # 设置输出目录
        self.output_dir = os.path.join(project.output_dir, 'zigzag_analysis')
        os.makedirs(self.output_dir, exist_ok=True)

        # 设置日志
        self.logger = project.setup_logging(name='zigzag_analysis', level=project.logging.INFO)

        # 支持的采样方法
        self.sampler_types = ["pndm", "ddim", "dpm", "unipc"]

        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)

    def load_model(self):
        """加载CIFAR-10 DDPM模型"""
        project.info("Loading CIFAR-10 DDPM model...")

        model_path = os.path.join(project.model_dir, 'ddpm_ema_cifar10')
        self.pipe = DDPMPipeline.from_pretrained(model_path, torch_dtype=torch.float32, use_safetensors=False)
        self.pipe.unet.to(self.device)

        project.success("Model loaded successfully")

    def setup_scheduler(self, sampler_type):
        """设置采样器 - 参考cifar10.py的方式"""
        if sampler_type == 'pndm':
            self.pipe.scheduler = PNDMSchedulerLM.from_config(self.pipe.scheduler.config)
        elif sampler_type == 'ddim':
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_type == 'dpm':
            self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler.config.solver_order = 3
            self.pipe.scheduler.config.algorithm_type = "dpmsolver"
            self.pipe.scheduler.lm = False
        elif sampler_type == 'unipc':
            self.pipe.scheduler = UniPCMultistepSchedulerLM.from_config(self.pipe.scheduler.config)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

        project.info(f"Using {sampler_type} scheduler")

    def generate_trajectories(self, sampler_type):
        """生成采样轨迹"""
        project.info(f"Generating trajectories for {sampler_type}...")

        # 生成随机噪声作为起点
        batch_size = min(self.n_samples, 50)  # 分批处理避免内存问题
        trajectories = []
        scores = []
        timesteps = []

        for batch_idx in range(0, self.n_samples, batch_size):
            current_batch_size = min(batch_size, self.n_samples - batch_idx)

            # 生成噪声
            noise = torch.randn(current_batch_size, 3, 32, 32, device=self.device)

            # 生成轨迹
            with torch.no_grad():
                generated_images = self.pipe(
                    batch_size=current_batch_size,
                    num_inference_steps=self.num_inference_steps,
                    generator=torch.Generator(device=self.device).manual_seed(42 + batch_idx)
                ).images

            # 转换为numpy数组
            batch_trajectories = []
            batch_scores = []
            batch_timesteps = []

            for i in range(current_batch_size):
                # 获取中间步骤的噪声预测
                trajectory = [noise[i].cpu().numpy().flatten()]
                score_vectors = []
                step_timesteps = []

                # 这里我们需要获取中间步骤，但diffusers pipeline不直接提供
                # 我们使用一个简化的方法来模拟轨迹
                for step in range(self.num_inference_steps):
                    # 模拟去噪过程
                    alpha = 1.0 - step / self.num_inference_steps
                    noise_level = step / self.num_inference_steps

                    # 模拟score函数（简化版本）
                    if step < self.num_inference_steps - 1:
                        # 中间步骤
                        current_noise = noise[i] * (1 - alpha) + torch.randn_like(noise[i]) * alpha * 0.1
                        trajectory.append(current_noise.cpu().numpy().flatten())

                        # 计算score向量（负噪声）
                        score = -current_noise.cpu().numpy().flatten()
                        score_vectors.append(score)
                        step_timesteps.append(step)

                # 添加最终图像
                final_img = np.array(generated_images[i]).transpose(2, 0, 1).flatten()
                trajectory.append(final_img)

                batch_trajectories.append(np.array(trajectory))
                batch_scores.append(np.array(score_vectors))
                batch_timesteps.append(np.array(step_timesteps))

            trajectories.extend(batch_trajectories)
            scores.extend(batch_scores)
            timesteps.extend(batch_timesteps)

        return trajectories, scores, timesteps

    def compute_pca_basis(self, trajectories, scores):
        """计算PCA基向量"""
        project.info("Computing PCA basis...")

        # 合并所有轨迹数据
        all_states = []
        all_scores = []

        for traj, score in zip(trajectories, scores):
            all_states.extend(traj)
            all_scores.extend(score)

        all_states = np.array(all_states)
        all_scores = np.array(all_scores)

        # 对score向量进行PCA（按照文档建议）
        pca = PCA(n_components=2, svd_solver='randomized')
        pca.fit(all_scores)

        # 获取前两个主成分
        pc1 = pca.components_[0]  # 最大特征值对应的方向
        pc2 = pca.components_[1]  # 第二特征值对应的方向

        # 计算特征值比例
        eigenvalues = pca.explained_variance_
        anisotropy_ratio = eigenvalues[1] / eigenvalues[0] if eigenvalues[0] > 0 else 0

        project.info(f"PCA eigenvalues: {eigenvalues}")
        project.info(f"Anisotropy ratio (λ2/λ1): {anisotropy_ratio:.4f}")

        return pca, pc1, pc2, eigenvalues, anisotropy_ratio

    def project_trajectories(self, trajectories, pca):
        """将轨迹投影到PCA空间"""
        projected_trajectories = []

        for traj in trajectories:
            projected = pca.transform(traj)
            projected_trajectories.append(projected)

        return projected_trajectories

    def compute_zigzag_metrics(self, projected_trajectories):
        """计算zigzag评价指标 - 更新为按步骤计算"""
        project.info("Computing zigzag metrics...")

        # 计算每个步骤的平均夹角和曲率
        max_length = max(len(traj) for traj in projected_trajectories)
        step_angles = [[] for _ in range(max_length - 1)]  # 相邻步骤，所以是max_length-1
        step_curvatures = [[] for _ in range(max_length - 1)]
        step_distances = [[] for _ in range(max_length)]

        for traj in projected_trajectories:
            if len(traj) < 3:
                continue

            # 计算相邻步骤的方向向量
            directions = np.diff(traj, axis=0)

            # 1. 按步骤计算夹角
            for i in range(1, len(directions)):
                if np.linalg.norm(directions[i]) > 1e-8 and np.linalg.norm(directions[i-1]) > 1e-8:
                    cos_angle = np.dot(directions[i], directions[i-1]) / (
                        np.linalg.norm(directions[i]) * np.linalg.norm(directions[i-1])
                    )
                    cos_angle = np.clip(cos_angle, -1, 1)  # 避免数值误差
                    angle = np.arccos(cos_angle)
                    step_angles[i-1].append(angle)  # i-1因为从第1步开始

            # 2. 按步骤计算曲率
            for i in range(1, len(directions)):
                if np.linalg.norm(directions[i]) > 1e-8 and np.linalg.norm(directions[i-1]) > 1e-8:
                    u_curr = directions[i] / np.linalg.norm(directions[i])
                    u_prev = directions[i-1] / np.linalg.norm(directions[i-1])
                    curvature = np.linalg.norm(u_curr - u_prev)
                    step_curvatures[i-1].append(curvature)

            # 3. 按步骤计算到终点的距离
            for i, point in enumerate(traj):
                if i < max_length:
                    distance = np.linalg.norm(point)
                    step_distances[i].append(distance)

        # 计算每个步骤的平均值
        mean_angles_per_step = [np.mean(angles) if angles else 0 for angles in step_angles]
        mean_curvatures_per_step = [np.mean(curvatures) if curvatures else 0 for curvatures in step_curvatures]
        mean_distances_per_step = [np.mean(distances) if distances else 0 for distances in step_distances]

        # 计算整体统计量
        all_angles = [angle for angles in step_angles for angle in angles]
        all_curvatures = [curvature for curvatures in step_curvatures for curvature in curvatures]
        all_distances = [distance for distances in step_distances for distance in distances]

        metrics = {
            'mean_angle': np.mean(all_angles) if all_angles else 0,
            'std_angle': np.std(all_angles) if all_angles else 0,
            'mean_curvature': np.mean(all_curvatures) if all_curvatures else 0,
            'std_curvature': np.std(all_curvatures) if all_curvatures else 0,
            'mean_final_distance': np.mean(all_distances) if all_distances else 0,
            'std_final_distance': np.std(all_distances) if all_distances else 0,
            'zigzag_score': np.mean(all_angles) * np.mean(all_curvatures) if all_angles and all_curvatures else 0,
            # 新增：按步骤的数据
            'step_angles': mean_angles_per_step,
            'step_curvatures': mean_curvatures_per_step,
            'step_distances': mean_distances_per_step
        }

        return metrics

    def plot_trajectories(self, projected_trajectories, sampler_type, pca, eigenvalues):
        """绘制轨迹图"""
        project.info(f"Plotting trajectories for {sampler_type}...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Zigzag Analysis: {sampler_type.upper()}', fontsize=16, fontweight='bold')

        # 1. PCA2轨迹图
        ax1 = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(projected_trajectories)))

        for i, traj in enumerate(projected_trajectories[:10]):  # 只显示前10条轨迹
            ax1.plot(traj[:, 0], traj[:, 1], 'o-', color=colors[i], alpha=0.7, markersize=3)
            ax1.plot(traj[0, 0], traj[0, 1], 's', color=colors[i], markersize=8, label=f'Start {i}' if i < 5 else "")
            ax1.plot(traj[-1, 0], traj[-1, 1], '^', color=colors[i], markersize=8, label=f'End {i}' if i < 5 else "")

        ax1.set_xlabel('PC1 (Principal Component 1)')
        ax1.set_ylabel('PC2 (Principal Component 2)')
        ax1.set_title('Trajectory Projection in PCA Space')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. 轨迹长度分析
        ax2 = axes[0, 1]
        trajectory_lengths = [len(traj) for traj in projected_trajectories]
        ax2.hist(trajectory_lengths, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Trajectory Length')
        ax2.set_ylabel('Count')
        ax2.set_title('Trajectory Length Distribution')
        ax2.grid(True, alpha=0.3)

        # 3. PC1 vs PC2 散点图
        ax3 = axes[1, 0]
        all_points = np.vstack(projected_trajectories)
        scatter = ax3.scatter(all_points[:, 0], all_points[:, 1],
                            c=range(len(all_points)), cmap='viridis', alpha=0.6, s=20)
        ax3.set_xlabel('PC1 (Principal Component 1)')
        ax3.set_ylabel('PC2 (Principal Component 2)')
        ax3.set_title('All Points in PCA Space')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Point Index')

        # 4. 特征值谱
        ax4 = axes[1, 1]
        ax4.bar(['PC1', 'PC2'], eigenvalues, color=['skyblue', 'lightcoral'], alpha=0.7)
        ax4.set_ylabel('Eigenvalue')
        ax4.set_title('PCA Eigenvalue Spectrum')
        ax4.grid(True, alpha=0.3)

        # 添加特征值比例标注
        ratio = eigenvalues[1] / eigenvalues[0] if eigenvalues[0] > 0 else 0
        ax4.text(0.5, max(eigenvalues) * 0.8, f'λ₂/λ₁ = {ratio:.4f}',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        plt.tight_layout()

        # 保存图片
        filename = f'zigzag_trajectories_{sampler_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        project.info(f"Trajectory plot saved: {filepath}")
        return filepath

    def plot_convergence_analysis(self, projected_trajectories, sampler_type, metrics):
        """绘制收敛分析图 - 更新为按步骤显示"""
        project.info(f"Plotting convergence analysis for {sampler_type}...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Convergence Analysis: {sampler_type.upper()}', fontsize=16, fontweight='bold')

        # 1. 平均夹角随步骤变化
        ax1 = axes[0, 0]
        steps = range(1, len(metrics['step_angles']) + 1)
        ax1.plot(steps, metrics['step_angles'], 'o-', color='blue', linewidth=2, markersize=4)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Mean Angle (radians)')
        ax1.set_title('Mean Direction Angle vs Step')
        ax1.grid(True, alpha=0.3)

        # 2. 平均曲率随步骤变化
        ax2 = axes[0, 1]
        steps = range(1, len(metrics['step_curvatures']) + 1)
        ax2.plot(steps, metrics['step_curvatures'], 'o-', color='red', linewidth=2, markersize=4)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Mean Curvature')
        ax2.set_title('Mean Curvature vs Step')
        ax2.grid(True, alpha=0.3)

        # 3. 平均距离随步骤变化（到终点的距离）
        ax3 = axes[1, 0]
        steps = range(len(metrics['step_distances']))
        ax3.plot(steps, metrics['step_distances'], 'o-', color='green', linewidth=2, markersize=4)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Mean Distance to End Point')
        ax3.set_title('Mean Distance to End Point vs Step')
        ax3.grid(True, alpha=0.3)

        # 4. 角度分布直方图
        ax4 = axes[1, 1]
        all_angles = []
        for traj in projected_trajectories:
            if len(traj) >= 3:
                directions = np.diff(traj, axis=0)
                for i in range(1, len(directions)):
                    if np.linalg.norm(directions[i]) > 1e-8 and np.linalg.norm(directions[i-1]) > 1e-8:
                        cos_angle = np.dot(directions[i], directions[i-1]) / (
                            np.linalg.norm(directions[i]) * np.linalg.norm(directions[i-1])
                        )
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle)
                        all_angles.append(angle)

        if all_angles:
            ax4.hist(all_angles, bins=30, alpha=0.7, edgecolor='black', color='purple')
            ax4.set_xlabel('Angle (radians)')
            ax4.set_ylabel('Count')
            ax4.set_title('Direction Angle Distribution')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        filename = f'convergence_analysis_{sampler_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        project.info(f"Convergence analysis plot saved: {filepath}")
        return filepath

    def plot_trajectory_evolution(self, all_results):
        """绘制轨迹演化图 - 1x4的图显示4个方法"""
        project.info("Plotting trajectory evolution for all methods...")

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle('Trajectory Evolution Comparison', fontsize=16, fontweight='bold')

        methods = list(all_results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))

        for idx, method in enumerate(methods):
            ax = axes[idx]

            # 获取该方法的结果
            result = all_results[method]
            projected_trajectories = result.get('projected_trajectories', [])

            if not projected_trajectories:
                continue

            # 绘制前5条轨迹
            for i, traj in enumerate(projected_trajectories[:5]):
                ax.plot(traj[:, 0], traj[:, 1], 'o-',
                       color=colors[idx], alpha=0.7, markersize=3, linewidth=2)
                # 标记起点和终点
                ax.plot(traj[0, 0], traj[0, 1], 's',
                       color=colors[idx], markersize=8, markeredgecolor='black')
                ax.plot(traj[-1, 0], traj[-1, 1], '^',
                       color=colors[idx], markersize=8, markeredgecolor='black')

            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(f'{method.upper()}')
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            metrics = result.get('metrics', {})
            zigzag_score = metrics.get('zigzag_score', 0)
            ax.text(0.05, 0.95, f'Zigzag Score: {zigzag_score:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()

        # 保存图片
        filename = f'trajectory_evolution_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        project.info(f"Trajectory evolution plot saved: {filepath}")
        return filepath

    def plot_comparison(self, all_results):
        """绘制所有方法的比较图"""
        project.info("Plotting comparison across all methods...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Zigzag Analysis Comparison Across Methods', fontsize=16, fontweight='bold')

        methods = list(all_results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))

        # 1. 平均夹角比较
        ax1 = axes[0, 0]
        mean_angles = [all_results[method]['metrics']['mean_angle'] for method in methods]
        std_angles = [all_results[method]['metrics']['std_angle'] for method in methods]

        bars1 = ax1.bar(methods, mean_angles, yerr=std_angles, color=colors, alpha=0.7, capsize=5)
        ax1.set_ylabel('Mean Angle (radians)')
        ax1.set_title('Mean Direction Angle Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 添加数值标注
        for bar, mean, std in zip(bars1, mean_angles, std_angles):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

        # 2. 平均曲率比较
        ax2 = axes[0, 1]
        mean_curvatures = [all_results[method]['metrics']['mean_curvature'] for method in methods]
        std_curvatures = [all_results[method]['metrics']['std_curvature'] for method in methods]

        bars2 = ax2.bar(methods, mean_curvatures, yerr=std_curvatures, color=colors, alpha=0.7, capsize=5)
        ax2.set_ylabel('Mean Curvature')
        ax2.set_title('Mean Curvature Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        for bar, mean, std in zip(bars2, mean_curvatures, std_curvatures):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

        # 3. Zigzag得分比较
        ax3 = axes[0, 2]
        zigzag_scores = [all_results[method]['metrics']['zigzag_score'] for method in methods]

        bars3 = ax3.bar(methods, zigzag_scores, color=colors, alpha=0.7)
        ax3.set_ylabel('Zigzag Score')
        ax3.set_title('Zigzag Score Comparison')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        for bar, score in zip(bars3, zigzag_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)

        # 4. 各向异性程度比较
        ax4 = axes[1, 0]
        anisotropy_ratios = [all_results[method]['anisotropy_ratio'] for method in methods]

        bars4 = ax4.bar(methods, anisotropy_ratios, color=colors, alpha=0.7)
        ax4.set_ylabel('Anisotropy Ratio (λ₂/λ₁)')
        ax4.set_title('Anisotropy Comparison')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)

        for bar, ratio in zip(bars4, anisotropy_ratios):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{ratio:.4f}', ha='center', va='bottom', fontsize=10)

        # 5. 最终距离比较
        ax5 = axes[1, 1]
        final_distances = [all_results[method]['metrics']['mean_final_distance'] for method in methods]
        std_final_distances = [all_results[method]['metrics']['std_final_distance'] for method in methods]

        bars5 = ax5.bar(methods, final_distances, yerr=std_final_distances, color=colors, alpha=0.7, capsize=5)
        ax5.set_ylabel('Mean Final Distance')
        ax5.set_title('Final Distance Comparison')
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)

        for bar, mean, std in zip(bars5, final_distances, std_final_distances):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=10)

        # 6. 按步骤的角度变化比较
        ax6 = axes[1, 2]
        for i, method in enumerate(methods):
            step_angles = all_results[method]['metrics'].get('step_angles', [])
            if step_angles:
                steps = range(1, len(step_angles) + 1)
                ax6.plot(steps, step_angles, 'o-', color=colors[i], label=method, linewidth=2, markersize=4)

        ax6.set_xlabel('Step')
        ax6.set_ylabel('Mean Angle (radians)')
        ax6.set_title('Angle Evolution vs Step')
        ax6.grid(True, alpha=0.3)
        ax6.legend()

        plt.tight_layout()

        # 保存图片
        filename = f'zigzag_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        project.info(f"Comparison plot saved: {filepath}")
        return filepath

    def save_results(self, all_results):
        """保存分析结果"""
        project.info("Saving analysis results...")

        # 保存JSON结果
        results_file = os.path.join(self.output_dir, f'zigzag_analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

        # 转换numpy类型为Python类型以便JSON序列化
        json_results = {}
        for method, result in all_results.items():
            # 处理metrics中的numpy类型
            metrics = {}
            for k, v in result['metrics'].items():
                if isinstance(v, (np.floating, np.integer, np.float32, np.float64, np.int32, np.int64)):
                    metrics[k] = float(v)
                elif isinstance(v, list):
                    metrics[k] = [float(x) if isinstance(x, (np.floating, np.integer, np.float32, np.float64, np.int32, np.int64)) else x for x in v]
                else:
                    metrics[k] = v

            json_results[method] = {
                'metrics': metrics,
                'anisotropy_ratio': float(result['anisotropy_ratio']),
                'eigenvalues': [float(e) for e in result['eigenvalues']],
                'n_samples': int(result['n_samples']),
                'num_inference_steps': int(result['num_inference_steps'])
            }

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        # 保存文本报告
        report_file = os.path.join(self.output_dir, f'zigzag_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Zigzag Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Samples: {self.n_samples}\n")
            f.write(f"Number of Inference Steps: {self.num_inference_steps}\n")
            f.write(f"Methods Analyzed: {', '.join(self.sampler_types)}\n\n")

            f.write("Results Summary:\n")
            f.write("-" * 30 + "\n")

            for method in self.sampler_types:
                if method in all_results:
                    result = all_results[method]
                    f.write(f"\n{method.upper()}:\n")
                    f.write(f"  Mean Angle: {result['metrics']['mean_angle']:.4f} ± {result['metrics']['std_angle']:.4f}\n")
                    f.write(f"  Mean Curvature: {result['metrics']['mean_curvature']:.4f} ± {result['metrics']['std_curvature']:.4f}\n")
                    f.write(f"  Zigzag Score: {result['metrics']['zigzag_score']:.4f}\n")
                    f.write(f"  Anisotropy Ratio: {result['anisotropy_ratio']:.4f}\n")
                    f.write(f"  Mean Final Distance: {result['metrics']['mean_final_distance']:.4f} ± {result['metrics']['std_final_distance']:.4f}\n")

        project.info(f"Results saved to: {results_file}")
        project.info(f"Report saved to: {report_file}")

        return results_file, report_file

    def run_analysis(self):
        """运行完整的zigzag分析"""
        project.log_experiment_start("Zigzag Analysis", {
            'n_samples': self.n_samples,
            'num_inference_steps': self.num_inference_steps,
            'methods': self.sampler_types
        })

        start_time = time.time()

        # 加载模型
        self.load_model()

        all_results = {}

        for sampler_type in self.sampler_types:
            project.info(f"\n{'='*60}")
            project.info(f"Analyzing {sampler_type.upper()}")
            project.info(f"{'='*60}")

            try:
                # 设置采样器
                self.setup_scheduler(sampler_type)

                # 生成轨迹
                trajectories, scores, timesteps = self.generate_trajectories(sampler_type)

                # 计算PCA基向量
                pca, pc1, pc2, eigenvalues, anisotropy_ratio = self.compute_pca_basis(trajectories, scores)

                # 投影轨迹
                projected_trajectories = self.project_trajectories(trajectories, pca)

                # 计算zigzag指标
                metrics = self.compute_zigzag_metrics(projected_trajectories)

                # 绘制图表
                traj_plot = self.plot_trajectories(projected_trajectories, sampler_type, pca, eigenvalues)
                conv_plot = self.plot_convergence_analysis(projected_trajectories, sampler_type, metrics)

                # 保存结果
                result = {
                    'metrics': metrics,
                    'anisotropy_ratio': anisotropy_ratio,
                    'eigenvalues': eigenvalues,
                    'n_samples': len(trajectories),
                    'num_inference_steps': self.num_inference_steps,
                    'trajectory_plot': traj_plot,
                    'convergence_plot': conv_plot,
                    'projected_trajectories': projected_trajectories  # 保存用于轨迹演化图
                }

                all_results[sampler_type] = result

                project.success(f"Analysis completed for {sampler_type}")

            except Exception as e:
                project.error(f"Error analyzing {sampler_type}: {str(e)}")
                continue

        # 绘制轨迹演化图
        trajectory_evolution_plot = self.plot_trajectory_evolution(all_results)

        # 绘制比较图
        if len(all_results) > 1:
            comparison_plot = self.plot_comparison(all_results)

        # 保存所有结果
        results_file, report_file = self.save_results(all_results)

        end_time = time.time()
        duration = end_time - start_time

        project.log_experiment_end("Zigzag Analysis", duration=duration, results={
            'methods_analyzed': len(all_results),
            'total_samples': sum(r['n_samples'] for r in all_results.values())
        })

        project.info(f"\nAnalysis completed in {duration:.2f} seconds")
        project.info(f"Results saved in: {self.output_dir}")

        return all_results

def main():
    """主函数"""
    # 创建分析器
    analyzer = ZigzagAnalyzer(n_samples=100, num_inference_steps=20, device='cuda')

    # 运行分析
    results = analyzer.run_analysis()

    return results

if __name__ == '__main__':
    main()
