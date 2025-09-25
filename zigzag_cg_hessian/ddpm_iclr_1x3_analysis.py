#!/usr/bin/env python3
"""
DDPM ICLR 1x3 Analysis: XT Space PCA2, PC2/PC1 Ratio, Score Space PCA2
专门针对DDPM方法的1×3 ICLR格式分析
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
from diffusers import DDPMPipeline, DDPMScheduler

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

class DDPMICLRAnalysis:
    """DDPM ICLR 1x3 Analysis"""

    def __init__(self, n_samples=5000, num_inference_steps=25, num_trajectories=50):
        self.n_samples = n_samples
        self.num_inference_steps = num_inference_steps
        self.num_trajectories = num_trajectories
        self.method = 'ddpm'

        print(f"🔧 DDPM ICLR 1x3 Analysis Configuration:")
        print(f"   - CIFAR-10 samples for PCA: {self.n_samples}")
        print(f"   - Inference steps per trajectory: {self.num_inference_steps}")
        print(f"   - Number of trajectories: {self.num_trajectories}")
        print(f"   - Method: {self.method.upper()}")

    def load_pipeline(self):
        """Load DDPM pipeline"""
        model_id = os.path.join(project.model_dir, 'ddpm_ema_cifar10')

        print(f"\n🔧 Loading {self.method.upper()} pipeline...")
        pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        pipe.unet.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup DDPM scheduler
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
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

            # Sampling step
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

    def plot_iclr_1x3_analysis(self, trajectories, xt_pca, score_pca, step_ratios, save_dir='./zigzag_cg_hessian'):
        """Plot ICLR 1x3 analysis for DDPM"""
        os.makedirs(save_dir, exist_ok=True)

        # Create figure with 1x3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle('DDPM Analysis: XT Space PCA2, PC2/PC1 Ratio, Score Space PCA2',
                     fontsize=16, fontweight='bold', y=0.95)

        # Define colors
        start_color = '#27AE60'  # Green
        end_color = '#E67E22'    # Orange
        method_color = '#E67E22'  # Orange for DDPM

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

        # Subplot 2: PC2/PC1 Ratio per Step
        ax2 = axes[1]
        steps = np.arange(self.num_inference_steps)
        ax2.plot(steps, step_ratios, 'o-', color=method_color, linewidth=3, markersize=6)
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Theoretical (1.0)')
        ax2.set_xlabel('Step', fontsize=12, fontweight='bold')
        interval = len(steps) // 5
        ax2.set_xticks(steps[::interval])
        ax2.set_xticklabels([str(int(x)) for x in steps[::interval]])  # 确保刻度标签为整数
        ax2.set_ylabel('PC2/PC1 Ratio', fontsize=12, fontweight='bold')
        ax2.set_title('PC2/PC1 Ratio per Step\n(Deviation from Theoretical)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Add statistics text
        # mean_ratio = np.mean(step_ratios)
        # std_ratio = np.std(step_ratios)
        # ax2.text(0.05, step_ratios[-1]+0.05, f'Mean: {mean_ratio:.4f}\nStd: {std_ratio:.4f}',
        #         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
        #         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

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

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'ddpm_iclr_1x3_analysis_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\n✓ DDPM ICLR 1x3 analysis plot saved to: {save_path}")

        plt.close()

    def generate_analysis_report(self, xt_pca, score_pca, step_ratios, save_dir='./zigzag_cg_hessian'):
        """Generate analysis report"""
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(save_dir, f'ddpm_iclr_1x3_analysis_report_{timestamp}.txt')

        with open(report_path, 'w') as f:
            f.write("DDPM ICLR 1x3 Analysis Report\n")
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

            f.write("STEP-BY-STEP RATIOS:\n")
            f.write("-" * 20 + "\n")
            for i, ratio in enumerate(step_ratios):
                f.write(f"Step {i:2d}: {ratio:.6f}\n")

        print(f"\n✓ Analysis report saved to: {report_path}")

        # Print summary to console
        print(f"\n" + "="*80)
        print("DDPM ICLR 1x3 ANALYSIS SUMMARY")
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

def main():
    """Main function to run DDPM ICLR 1x3 analysis"""

    print("🚀 DDPM ICLR 1x3 Analysis")
    print("="*50)
    print("XT Space PCA2, PC2/PC1 Ratio, Score Space PCA2")
    print("="*50)

    # Initialize analyzer
    analyzer = DDPMICLRAnalysis(n_samples=10000, num_inference_steps=500, num_trajectories=200)

    try:
        # Load pipeline
        print(f"\n{'='*60}")
        print("Loading DDPM Pipeline")
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

        # Calculate PC2/PC1 ratio per step
        print(f"\n{'='*60}")
        print("Calculating PC2/PC1 Ratio per Step")
        print(f"{'='*60}")
        step_ratios = analyzer.calculate_pc2_pc1_ratio_per_step(trajectories, xt_pca)

        # Create ICLR 1x3 analysis plots
        print(f"\n{'='*60}")
        print("Creating ICLR 1x3 Analysis Plots")
        print(f"{'='*60}")
        analyzer.plot_iclr_1x3_analysis(trajectories, xt_pca, score_pca, step_ratios)

        # Generate analysis report
        print(f"\n{'='*60}")
        print("Generating Analysis Report")
        print(f"{'='*60}")
        analyzer.generate_analysis_report(xt_pca, score_pca, step_ratios)

        print(f"\n✅ DDPM ICLR 1x3 analysis completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
