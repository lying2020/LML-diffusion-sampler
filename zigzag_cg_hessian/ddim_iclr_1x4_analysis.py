#!/usr/bin/env python3
"""
DDIM ICLR 1x4 Analysis: XT Space PCA2, PC2/PC1 Ratio, Score Space PCA2, Score Angle Analysis
专门针对DDIM方法的1×4 ICLR格式分析
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set matplotlib to use a backend that doesn't require display
import matplotlib
matplotlib.use('Agg')

# Import schedulers
from diffusers import DDPMPipeline, DDIMScheduler

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
    'lines.linewidth': 3,
    'lines.markersize': 6,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3
})

class DDIMICLRAnalysis:
    """DDIM ICLR 1x4 Analysis"""

    def __init__(self, n_samples=5000, num_inference_steps=25, num_trajectories=50):
        self.n_samples = n_samples
        self.num_inference_steps = num_inference_steps
        self.num_trajectories = num_trajectories
        self.method = 'ddim'

        print(f"🔧 DDIM ICLR 1x4 Analysis Configuration:")
        print(f"   - CIFAR-10 samples for PCA: {self.n_samples}")
        print(f"   - Inference steps per trajectory: {self.num_inference_steps}")
        print(f"   - Number of trajectories: {self.num_trajectories}")
        print(f"   - Method: {self.method.upper()}")

    def load_pipeline(self):
        """Load DDIM pipeline"""
        model_id = os.path.join(project.model_dir, 'ddpm_ema_cifar10')

        print(f"\n🔧 Loading {self.method.upper()} pipeline...")
        pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        pipe.unet.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup DDIM scheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(self.num_inference_steps)

        print(f"✓ {self.method.upper()} pipeline loaded successfully")
        return pipe

    def generate_trajectories(self, pipe, seed=42):
        """Generate multiple trajectories for statistical analysis"""
        print(f"\n🚀 Generating {self.num_trajectories} trajectories using {self.method.upper()}...")

        trajectories = []
        torch.manual_seed(seed)

        for i in range(self.num_trajectories):
            if i % 10 == 0:
                print(f"  Generating trajectory {i+1}/{self.num_trajectories}")

            trajectory_data = self._generate_single_trajectory(pipe, seed + i)
            trajectories.append(trajectory_data)

        print(f"✓ Generated {len(trajectories)} trajectories")
        return trajectories

    def _generate_single_trajectory(self, pipe, seed):
        """Generate a single trajectory and collect intermediate states"""
        torch.manual_seed(seed)

        # Initialize random noise
        device = pipe.unet.device
        shape = (1, pipe.unet.config.in_channels, 32, 32)
        latents = torch.randn(shape, device=device)

        # Store trajectory data
        trajectory_data = {
            'xt': [],      # 待生成的图片状态向量
            'score': [],   # 分数/漂移向量
            'timesteps': [],
            'noise_pred': []
        }

        scheduler = pipe.scheduler

        for j, t in enumerate(scheduler.timesteps):
            # Store current state
            xt_flat = latents.detach().view(1, -1).cpu().numpy().flatten()
            trajectory_data['xt'].append(xt_flat)
            trajectory_data['timesteps'].append(t.item())

            # Predict noise
            with torch.no_grad():
                noise_pred = pipe.unet(latents, t).sample

            # Store noise prediction
            noise_pred_flat = noise_pred.detach().view(1, -1).cpu().numpy().flatten()
            trajectory_data['noise_pred'].append(noise_pred_flat)

            # Compute score/drift vector
            score = -noise_pred_flat
            trajectory_data['score'].append(score)

            # DDIM sampling step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Convert to numpy arrays
        for key in ['xt', 'score', 'timesteps', 'noise_pred']:
            trajectory_data[key] = np.array(trajectory_data[key])

        return trajectory_data

    def create_pca_models(self, trajectories):
        """Create PCA models for both XT and Score analysis"""
        print(f"\n📊 Creating PCA models...")

        # Collect XT data
        xt_data = []
        for traj in trajectories:
            xt_data.append(traj['xt'])
        xt_data = np.vstack(xt_data)
        print(f"  XT data shape: {xt_data.shape}")

        # Collect Score data
        score_data = []
        for traj in trajectories:
            score_data.append(traj['score'])
        score_data = np.vstack(score_data)
        print(f"  Score data shape: {score_data.shape}")

        # Create PCA models
        xt_pca = PCA(n_components=2, svd_solver='randomized')
        xt_pca.fit(xt_data)
        print(f"  XT PCA - PC1: {xt_pca.explained_variance_ratio_[0]:.4f}, PC2: {xt_pca.explained_variance_ratio_[1]:.4f}")

        score_pca = PCA(n_components=2, svd_solver='randomized')
        score_pca.fit(score_data)
        print(f"  Score PCA - PC1: {score_pca.explained_variance_ratio_[0]:.4f}, PC2: {score_pca.explained_variance_ratio_[1]:.4f}")

        return xt_pca, score_pca, xt_data, score_data

    def calculate_pc2_pc1_ratio_per_step(self, trajectories, xt_pca):
        """Calculate PC2/PC1 ratio for each step"""
        print(f"\n📊 Calculating PC2/PC1 ratio per step...")

        # Collect all step data
        step_ratios = []
        for step in range(self.num_inference_steps):
            step_data = []
            for traj in trajectories:
                step_data.append(traj['xt'][step])

            step_data = np.array(step_data)

            # Calculate PCA for this step
            step_pca = PCA(n_components=2, svd_solver='randomized')
            step_pca.fit(step_data)

            # Calculate PC2/PC1 ratio
            pc1_var = step_pca.explained_variance_[0]
            pc2_var = step_pca.explained_variance_[1]
            ratio = pc2_var / pc1_var if pc1_var > 0 else 0

            step_ratios.append(ratio)

            if step % 5 == 0:
                print(f"  Step {step}: PC2/PC1 = {ratio:.6f}")

        return np.array(step_ratios)


    def calculate_score_angles(self, trajectories):
        """Calculate angles between consecutive score vectors"""
        print(f"\n📊 Calculating score vector angles...")
        
        all_angles = []
        for traj in trajectories:
            score_vectors = traj['score']  # [T, D]
            angles = []
            
            for i in range(1, len(score_vectors)):
                # 计算score_i和score_{i-1}的夹角
                score_curr = score_vectors[i]
                score_prev = score_vectors[i-1]
                
                # 计算余弦相似度
                dot_product = np.dot(score_curr, score_prev)
                norm_curr = np.linalg.norm(score_curr)
                norm_prev = np.linalg.norm(score_prev)
                
                if norm_curr > 0 and norm_prev > 0:
                    cos_angle = dot_product / (norm_curr * norm_prev)
                    # 限制在[-1, 1]范围内，避免数值误差
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
                else:
                    angles.append(0.0)
            
            all_angles.append(angles)
        
        # 计算平均角度
        mean_angles = np.mean(all_angles, axis=0)
        print(f"  Score angles calculated for {len(trajectories)} trajectories")
        print(f"  Mean angle range: {np.min(mean_angles):.4f} - {np.max(mean_angles):.4f} radians")
        
        return mean_angles

    def find_high_slope_points(self, step_ratios, num_points=3):
        """Find points with high slope changes in PC2/PC1 ratio, one from each third of the trajectory"""
        # Calculate slope (first derivative)
        slopes = np.diff(step_ratios)

        # Divide trajectory into 3 equal parts
        total_steps = len(step_ratios)
        part_size = total_steps // 3

        # Define three intervals: early, middle, late
        intervals = [
            (0, part_size),
            (part_size, 2 * part_size),
            (2 * part_size, total_steps)
        ]

        # Find the point with highest slope in each interval
        selected_points = []
        for i, (start, end) in enumerate(intervals):
            # Ensure we have enough space for 7-point window (3 points on each side)
            interval_start = max(start, 3)
            interval_end = min(end, total_steps - 4)

            if interval_start < interval_end:
                # Get slopes in this interval
                interval_slopes = slopes[interval_start:interval_end]
                if len(interval_slopes) > 0:
                    # Find the index with highest absolute slope in this interval
                    local_max_idx = np.argmax(np.abs(interval_slopes))
                    global_idx = interval_start + local_max_idx
                    selected_points.append(global_idx)
                    print(f"  Interval {i+1} [{interval_start}:{interval_end}]: selected t={global_idx}, slope={slopes[global_idx]:.6f}")

        return selected_points[:num_points]

    def plot_local_window(self, Z2d, t, ax, title):
        """
        在给定 Axes 上绘制 [t-3, t-2, t-1, t, t+1, t+2, t+3] 的投影折线
        Z2d: [T, 2] 的投影坐标
        """
        assert 3 <= t <= Z2d.shape[0]-4, f"t={t} 超出可取 7 帧窗口的范围"
        idx = np.arange(t-3, t+4)
        pts = Z2d[idx]  # [7, 2]

        ax.plot(pts[:, 0], pts[:, 1], "-o", linewidth=2, markersize=5)
        # 用方块标注中心帧 t
        ax.scatter(pts[3, 0], pts[3, 1], s=120, marker='s', zorder=3, label=f"t={t}")
        ax.legend(frameon=False, loc="best")
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.axis("equal")
        ax.grid(True, linestyle=":")

    def plot_iclr_1x4_analysis(self, trajectories, xt_pca, score_pca, step_ratios, score_angles, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """Plot ICLR 1x4 analysis for DDIM"""
        os.makedirs(save_dir, exist_ok=True)

        # Create figure with 1x4 subplots
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle('DDIM Analysis: XT Space PCA2, PC2/PC1 Ratio, Score Space PCA2, Score Angle Analysis',
                     fontsize=16, fontweight='bold', y=0.95)

        # Define colors
        start_color = '#27AE60'  # Green
        end_color = '#E67E22'    # Orange
        method_color = '#3498DB'  # Blue for DDIM

        # Subplot 1: XT Space PCA2 Analysis
        ax1 = axes[0]
        traj = trajectories[0]  # Use first trajectory
        xt_pca_proj = xt_pca.transform(traj['xt'])
        n_points = len(xt_pca_proj)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))

        # Plot trajectory
        for j in range(n_points - 1):
            ax1.plot([xt_pca_proj[j, 0], xt_pca_proj[j+1, 0]],
                   [xt_pca_proj[j, 1], xt_pca_proj[j+1, 1]],
                   color=colors[j], linewidth=3, alpha=0.9)

        # Mark start and end points
        ax1.scatter(xt_pca_proj[0, 0], xt_pca_proj[0, 1],
                  c=start_color, s=200, marker='o', label='Start', zorder=5,
                  edgecolors='black', linewidth=2)
        ax1.scatter(xt_pca_proj[-1, 0], xt_pca_proj[-1, 1],
                  c=end_color, s=200, marker='s', label='End', zorder=5,
                  edgecolors='black', linewidth=2)

        # Add step markers
        for j in range(0, n_points, max(1, n_points//8)):
            ax1.scatter(xt_pca_proj[j, 0], xt_pca_proj[j, 1],
                      c=colors[j], s=80, marker='o', alpha=0.8, zorder=3)

        ax1.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax1.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax1.set_title('XT Space PCA2 Analysis\n(Image State Evolution)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # Subplot 2: PC2/PC1 Ratio per Step (最多显示40个点)
        ax2 = axes[1]
        steps = np.arange(self.num_inference_steps)

        # 如果步数超过40，则均匀采样40个点
        if len(steps) > 40:
            sample_indices = np.linspace(0, len(steps)-1, 40, dtype=int)
            sampled_steps = steps[sample_indices]
            sampled_ratios = step_ratios[sample_indices]
        else:
            sampled_steps = steps
            sampled_ratios = step_ratios

        ax2.plot(sampled_steps, sampled_ratios, 'o-', color=method_color, linewidth=3, markersize=6)
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Theoretical (1.0)')
        ax2.set_xlabel('Step', fontsize=12, fontweight='bold')
        # 设置x轴刻度 - 显示整百或整十的刻度
        if len(steps) > 100:
            # 对于大范围，使用整百刻度
            max_step = steps[-1]
            if max_step >= 1000:
                tick_interval = 200
            elif max_step >= 500:
                tick_interval = 100
            else:
                tick_interval = 50
            
            # 生成整百/整十刻度
            tick_steps = np.arange(0, max_step + 1, tick_interval)
            # 确保包含最后一个点
            if tick_steps[-1] < max_step:
                tick_steps = np.append(tick_steps, max_step)
            
            ax2.set_xticks(tick_steps)
            ax2.set_xticklabels([str(int(x)) for x in tick_steps])
        elif len(steps) > 10:
            # 对于中等范围，使用整十刻度
            max_step = steps[-1]
            tick_interval = max(10, max_step // 10)
            tick_steps = np.arange(0, max_step + 1, tick_interval)
            if tick_steps[-1] < max_step:
                tick_steps = np.append(tick_steps, max_step)
            
            ax2.set_xticks(tick_steps)
            ax2.set_xticklabels([str(int(x)) for x in tick_steps])
        else:
            # 对于小范围，显示所有刻度
            ax2.set_xticks(steps)
            ax2.set_xticklabels([str(int(x)) for x in steps])
            # 对于小范围，显示所有刻度
            ax2.set_xticks(steps)
            ax2.set_xticklabels([str(int(x)) for x in steps])

        ax2.set_ylabel('PC2/PC1 Ratio', fontsize=12, fontweight='bold')
        ax2.set_title('PC2/PC1 Ratio per Step\n(Deviation from Theoretical)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Score Space PCA2 Analysis
        ax3 = axes[2]
        traj = trajectories[0]  # Use first trajectory
        score_pca_proj = score_pca.transform(traj['score'])
        n_points = len(score_pca_proj)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))

        # Plot trajectory
        for j in range(n_points - 1):
            ax3.plot([score_pca_proj[j, 0], score_pca_proj[j+1, 0]],
                   [score_pca_proj[j, 1], score_pca_proj[j+1, 1]],
                   color=colors[j], linewidth=3, alpha=0.9)

        # Mark start and end points
        ax3.scatter(score_pca_proj[0, 0], score_pca_proj[0, 1],
                  c=start_color, s=200, marker='o', label='Start', zorder=5,
                  edgecolors='black', linewidth=2)
        ax3.scatter(score_pca_proj[-1, 0], score_pca_proj[-1, 1],
                  c=end_color, s=200, marker='s', label='End', zorder=5,
                  edgecolors='black', linewidth=2)

        # Add step markers
        for j in range(0, n_points, max(1, n_points//8)):
            ax3.scatter(score_pca_proj[j, 0], score_pca_proj[j, 1],
                      c=colors[j], s=80, marker='o', alpha=0.8, zorder=3)

        ax3.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax3.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax3.set_title('Score Space PCA2 Analysis\n(Gradient Evolution)', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10, loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')

        # Subplot 4: Score Vector Angle Analysis
        ax4 = axes[3]
        steps = np.arange(1, self.num_inference_steps)  # 角度从第2步开始
        
        # 如果步数超过40，则均匀采样40个点
        if len(steps) > 40:
            sample_indices = np.linspace(0, len(steps)-1, 40, dtype=int)
            sampled_steps = steps[sample_indices]
            sampled_angles = score_angles[sample_indices]
        else:
            sampled_steps = steps
            sampled_angles = score_angles

        ax4.plot(sampled_steps, sampled_angles, 'o-', color='#E74C3C', linewidth=3, markersize=6)
        ax4.axhline(y=np.pi/2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='90° (π/2)')
        # 设置x轴刻度 - 显示整百或整十的刻度
        if len(steps) > 100:
            # 对于大范围，使用整百刻度
            max_step = steps[-1]
            if max_step >= 1000:
                tick_interval = 200
            elif max_step >= 500:
                tick_interval = 100
            else:
                tick_interval = 50
            
            # 生成整百/整十刻度
            tick_steps = np.arange(0, max_step + 1, tick_interval)
            # 确保包含最后一个点
            if tick_steps[-1] < max_step:
                tick_steps = np.append(tick_steps, max_step)
            
            ax4.set_xticks(tick_steps)
            ax4.set_xticklabels([str(int(x)) for x in tick_steps])
        elif len(steps) > 10:
            # 对于中等范围，使用整十刻度
            max_step = steps[-1]
            tick_interval = max(10, max_step // 10)
            tick_steps = np.arange(0, max_step + 1, tick_interval)
            if tick_steps[-1] < max_step:
                tick_steps = np.append(tick_steps, max_step)
            
            ax4.set_xticks(tick_steps)
            ax4.set_xticklabels([str(int(x)) for x in tick_steps])
        else:
            # 对于小范围，显示所有刻度
            ax4.set_xticks(steps)
            ax4.set_xticklabels([str(int(x)) for x in steps])
        
        ax4.set_ylabel('Angle (radians)', fontsize=12, fontweight='bold')
        ax4.set_title('Score Vector Angle Analysis\n(Consecutive Score Angles)', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'ddim_iclr_1x4_analysis_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\n✓ DDIM ICLR 1x4 analysis plot saved to: {save_path}")

        plt.close()

        # 生成局部轨迹图
        self.plot_local_trajectory_windows(trajectories, xt_pca, step_ratios, save_dir)

    def plot_local_trajectory_windows(self, trajectories, xt_pca, step_ratios, save_dir):
        """Generate local trajectory windows at high slope points"""
        print(f"\n📊 Generating local trajectory windows...")

        # 找到高斜率变化点
        high_slope_points = self.find_high_slope_points(step_ratios, num_points=3)
        print(f"  High slope points: {high_slope_points}")

        # 使用第一条轨迹进行局部窗口分析
        traj = trajectories[0]
        xt_pca_proj = xt_pca.transform(traj['xt'])

        # 创建局部窗口图
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle('DDIM Local Trajectory Windows (7-point around high slope changes)',
                     fontsize=14, fontweight='bold', y=0.95)

        for i, t in enumerate(high_slope_points):
            ax = axes[i]
            self.plot_local_window(xt_pca_proj, t, ax, f"Local 7-step around t={t}\n(Slope: {np.diff(step_ratios)[t-1]:.4f})")

        plt.tight_layout()

        # 保存局部窗口图
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        local_windows_path = os.path.join(save_dir, f'ddim_local_windows_{timestamp}.png')
        plt.savefig(local_windows_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ DDIM local trajectory windows plot saved to: {local_windows_path}")

        plt.close()

    def generate_analysis_report(self, xt_pca, score_pca, step_ratios, score_angles, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """Generate analysis report"""
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(save_dir, f'ddim_iclr_1x4_analysis_report_{timestamp}.txt')

        with open(report_path, 'w') as f:
            f.write("DDIM ICLR 1x4 Analysis Report\n")
            f.write("="*50 + "\n\n")

            f.write("EXPERIMENTAL CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Method: {self.method.upper()}\n")
            f.write(f"Trajectories: {self.num_trajectories}\n")
            f.write(f"Inference steps per trajectory: {self.num_inference_steps}\n")
            f.write(f"Total data points: {self.num_trajectories * self.num_inference_steps}\n\n")

            f.write("XT SPACE PCA2 ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"PC1 explained variance: {xt_pca.explained_variance_ratio_[0]:.6f}\n")
            f.write(f"PC2 explained variance: {xt_pca.explained_variance_ratio_[1]:.6f}\n")
            f.write(f"Total explained variance: {np.sum(xt_pca.explained_variance_ratio_):.6f}\n")
            f.write(f"PC1/PC2 ratio: {xt_pca.explained_variance_[0]/xt_pca.explained_variance_[1]:.6f}\n\n")

            f.write("SCORE SPACE PCA2 ANALYSIS:\n")
            f.write("-" * 28 + "\n")
            f.write(f"PC1 explained variance: {score_pca.explained_variance_ratio_[0]:.6f}\n")
            f.write(f"PC2 explained variance: {score_pca.explained_variance_ratio_[1]:.6f}\n")
            f.write(f"Total explained variance: {np.sum(score_pca.explained_variance_ratio_):.6f}\n")
            f.write(f"PC1/PC2 ratio: {score_pca.explained_variance_[0]/score_pca.explained_variance_[1]:.6f}\n\n")

            f.write("PC2/PC1 RATIO PER STEP ANALYSIS:\n")
            f.write("-" * 32 + "\n")
            f.write(f"Mean ratio: {np.mean(step_ratios):.6f}\n")
            f.write(f"Std ratio: {np.std(step_ratios):.6f}\n")
            f.write(f"Min ratio: {np.min(step_ratios):.6f}\n")
            f.write(f"Max ratio: {np.max(step_ratios):.6f}\n")
            f.write(f"Deviation from 1.0: {np.mean(np.abs(step_ratios - 1.0)):.6f}\n\n")

            f.write("SCORE VECTOR ANGLE ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean angle: {np.mean(score_angles):.6f} radians ({np.degrees(np.mean(score_angles)):.2f}°)\n")
            f.write(f"Std angle: {np.std(score_angles):.6f} radians ({np.degrees(np.std(score_angles)):.2f}°)\n")
            f.write(f"Min angle: {np.min(score_angles):.6f} radians ({np.degrees(np.min(score_angles)):.2f}°)\n")
            f.write(f"Max angle: {np.max(score_angles):.6f} radians ({np.degrees(np.max(score_angles)):.2f}°)\n")
            f.write(f"Deviation from 90°: {np.mean(np.abs(score_angles - np.pi/2)):.6f} radians ({np.degrees(np.mean(np.abs(score_angles - np.pi/2))):.2f}°)\n\n")

            # 添加高斜率点信息
            high_slope_points = self.find_high_slope_points(step_ratios, num_points=3)
            f.write("HIGH SLOPE POINTS FOR LOCAL WINDOWS:\n")
            f.write("-" * 35 + "\n")
            for i, t in enumerate(high_slope_points):
                slope = np.diff(step_ratios)[t-1] if t > 0 else 0
                f.write(f"Point {i+1}: t={t}, slope={slope:.6f}\n")
            f.write("\n")

            f.write("STEP-BY-STEP RATIOS:\n")
            f.write("-" * 20 + "\n")
            for i, ratio in enumerate(step_ratios):
                f.write(f"Step {i:2d}: {ratio:.6f}\n")

        print(f"\n✓ Analysis report saved to: {report_path}")

        # Print summary to console
        print(f"\n" + "="*80)
        print("DDIM ICLR 1x4 ANALYSIS SUMMARY")
        print("="*80)

        print(f"\nXT Space PCA2:")
        print(f"PC1 explained variance: {xt_pca.explained_variance_ratio_[0]:.6f}")
        print(f"PC2 explained variance: {xt_pca.explained_variance_ratio_[1]:.6f}")
        print(f"PC1/PC2 ratio: {xt_pca.explained_variance_[0]/xt_pca.explained_variance_[1]:.6f}")

        print(f"\nScore Space PCA2:")
        print(f"PC1 explained variance: {score_pca.explained_variance_ratio_[0]:.6f}")
        print(f"PC2 explained variance: {score_pca.explained_variance_ratio_[1]:.6f}")
        print(f"PC1/PC2 ratio: {score_pca.explained_variance_[0]/score_pca.explained_variance_[1]:.6f}")

        print(f"\nPC2/PC1 Ratio per Step:")
        print(f"Mean: {np.mean(step_ratios):.6f}")
        print(f"Std: {np.std(step_ratios):.6f}")
        print(f"Deviation from 1.0: {np.mean(np.abs(step_ratios - 1.0)):.6f}")

        print(f"\nScore Vector Angles:")
        print(f"Mean: {np.mean(score_angles):.6f} radians ({np.degrees(np.mean(score_angles)):.2f}°)")
        print(f"Std: {np.std(score_angles):.6f} radians ({np.degrees(np.std(score_angles)):.2f}°)")
        print(f"Deviation from 90°: {np.mean(np.abs(score_angles - np.pi/2)):.6f} radians ({np.degrees(np.mean(np.abs(score_angles - np.pi/2))):.2f}°)")

        # 打印高斜率点信息
        high_slope_points = self.find_high_slope_points(step_ratios, num_points=3)
        print(f"\nHigh Slope Points for Local Windows:")
        for i, t in enumerate(high_slope_points):
            slope = np.diff(step_ratios)[t-1] if t > 0 else 0
            print(f"  Point {i+1}: t={t}, slope={slope:.6f}")

def main():
    """Main function to run DDIM ICLR 1x3 analysis"""

    print("🚀 DDIM ICLR 1x4 Analysis")
    print("="*50)
    print("XT Space PCA2, PC2/PC1 Ratio, Score Space PCA2")
    print("="*50)

    # Initialize analyzer
    analyzer = DDIMICLRAnalysis(n_samples=10000, num_inference_steps=500, num_trajectories=20)

    try:
        # Load pipeline
        print(f"\n{'='*60}")
        print("Loading DDIM Pipeline")
        print(f"{'='*60}")
        pipe = analyzer.load_pipeline()

        # Generate trajectories
        print(f"\n{'='*60}")
        print("Generating Trajectories")
        print(f"{'='*60}")
        trajectories = analyzer.generate_trajectories(pipe, seed=42)

        # Create PCA models
        print(f"\n{'='*60}")
        print("Creating PCA Models")
        print(f"{'='*60}")
        xt_pca, score_pca, xt_data, score_data = analyzer.create_pca_models(trajectories)

        print("Calculating PC2/PC1 Ratio per Step")
        print("="*60)
        step_ratios = analyzer.calculate_pc2_pc1_ratio_per_step(trajectories, xt_pca)

        # Calculate score vector angles
        print("="*60)
        print("Calculating Score Vector Angles")
        print("="*60)
        score_angles = analyzer.calculate_score_angles(trajectories)

        # Create ICLR 1x4 analysis plots
        print("\n" + "="*60)
        print("Creating ICLR 1x4 Analysis Plots")
        print("="*60)
        analyzer.plot_iclr_1x4_analysis(trajectories, xt_pca, score_pca, step_ratios, score_angles)

        # Generate analysis report
        print(f"\n{'='*60}")
        print("Generating Analysis Report")
        print(f"{'='*60}")
        analyzer.generate_analysis_report(xt_pca, score_pca, step_ratios, score_angles)

        print(f"\n✅ DDIM ICLR 1x4 analysis completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
