#!/usr/bin/env python3
"""
ICLR Paper Format: xt vs score PCA2 Analysis Comparison
比较xt和score两种分析对象的PCA2结果
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
from diffusers import DDPMPipeline, DDIMScheduler, DPMSolverMultistepScheduler, DDPMScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
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
    'lines.linewidth': 4,
    'lines.markersize': 8,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3
})

class XTvsScoreAnalysis:
    """比较xt和score两种分析对象的PCA2结果"""

    def __init__(self, n_samples=5000, num_inference_steps=25, num_trajectories=20):
        self.n_samples = n_samples
        self.num_inference_steps = num_inference_steps
        self.num_trajectories = num_trajectories
        self.methods = ['ddim', 'pndm', "dpm", "ddpm"]

        print(f"🔧 XT vs Score PCA2 Analysis Configuration:")
        print(f"   - CIFAR-10 samples for PCA: {self.n_samples}")
        print(f"   - Inference steps per trajectory: {self.num_inference_steps}")
        print(f"   - Number of trajectories per method: {self.num_trajectories}")
        print(f"   - Methods: {', '.join(self.methods)}")

    def load_pipeline(self, method_name):
        """Load pipeline for different sampling methods"""
        model_path = os.path.join(project.model_dir, 'ddpm_ema_cifar10')

        print(f"\n🔧 Loading {method_name.upper()} pipeline...")
        pipe = DDPMPipeline.from_pretrained(model_path, torch_dtype=torch.float32, use_safetensors=False)
        pipe.unet.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup scheduler based on method
        if method_name == 'ddim':
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif method_name == 'pndm':
            pipe.scheduler = PNDMSchedulerLM.from_config(pipe.scheduler.config)
        elif method_name == 'dpm':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lm = False
        elif method_name == 'ddpm':
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        # Set timesteps for ALL schedulers
        pipe.scheduler.set_timesteps(self.num_inference_steps)
        print(f"✓ {method_name.upper()} pipeline loaded successfully")
        return pipe

    def generate_trajectories(self, pipe, method_name, seed=42):
        """Generate multiple trajectories for statistical analysis"""
        print(f"\n🚀 Generating {self.num_trajectories} trajectories using {method_name.upper()}...")

        trajectories = []
        torch.manual_seed(seed)

        for i in range(self.num_trajectories):
            if i % 5 == 0:
                print(f"  Generating trajectory {i+1}/{self.num_trajectories}")

            trajectory_data = self._generate_single_trajectory(pipe, method_name, seed + i)
            trajectories.append(trajectory_data)

        print(f"✓ Generated {len(trajectories)} trajectories")
        return trajectories

    def _generate_single_trajectory(self, pipe, method_name, seed):
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

    def create_pca_from_trajectories(self, trajectories, analysis_type='xt'):
        """Create PCA model from trajectory data with different analysis types"""
        print(f"\n📊 Creating PCA model from {analysis_type} data...")

        all_trajectory_data = []
        for method, traj_list in trajectories.items():
            for traj in traj_list:
                if analysis_type == 'xt':
                    all_trajectory_data.append(traj['xt'])
                elif analysis_type == 'score':
                    all_trajectory_data.append(traj['score'])
                elif analysis_type == 'noise_pred':
                    all_trajectory_data.append(traj['noise_pred'])

        all_data = np.vstack(all_trajectory_data)
        print(f"  Total {analysis_type} data shape: {all_data.shape}")

        # Perform PCA
        pca_model = PCA(n_components=2, svd_solver='randomized')
        pca_model.fit(all_data)

        print(f"✓ PCA completed for {analysis_type}. Explained variance: {pca_model.explained_variance_ratio_}")
        return pca_model

    def plot_xt_vs_score_comparison(self, trajectories, xt_pca, score_pca, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """Plot comparison between xt and score analysis"""
        os.makedirs(save_dir, exist_ok=True)

        # Create figure with 2x4 subplots
        fig, axes = plt.subplots(4, 2, figsize=(20, 24))
        fig.suptitle('XT vs Score PCA2 Analysis Comparison\n(Trajectory Evolution in Different Spaces)',
                     fontsize=18, fontweight='bold', y=0.95)

        # Define colors for methods
        method_colors = {
            'ddim': '#E74C3C',      # Red
            'pndm': '#9B59B6',      # Purple
            'dpm': '#27AE60',      # Green
            'ddpm': '#E67E22',     # Orange
        }

        method_names = {
            'ddim': 'DDIM',
            'pndm': 'PNDM',
            'dpm': 'DPM',
            'ddpm': 'DDPM',
        }

        start_color = '#27AE60'  # Green
        end_color = '#E67E22'    # Orange

        # Plot XT analysis (top row)
        for i, method in enumerate(self.methods):
            ax = axes[i, 0]
            traj = trajectories[method][0]  # Use first trajectory
            method_color = method_colors[method]
            method_name = method_names[method]

            xt_pca_proj = xt_pca.transform(traj['xt'])
            n_points = len(xt_pca_proj)
            colors = plt.cm.viridis(np.linspace(0, 1, n_points))

            # Plot trajectory
            for j in range(n_points - 1):
                ax.plot([xt_pca_proj[j, 0], xt_pca_proj[j+1, 0]],
                       [xt_pca_proj[j, 1], xt_pca_proj[j+1, 1]],
                       color=colors[j], linewidth=3, alpha=0.9)

            # Mark start and end points
            ax.scatter(xt_pca_proj[0, 0], xt_pca_proj[0, 1],
                      c=start_color, s=150, marker='o', label='Start', zorder=5,
                      edgecolors='black', linewidth=2)
            ax.scatter(xt_pca_proj[-1, 0], xt_pca_proj[-1, 1],
                      c=end_color, s=150, marker='s', label='End', zorder=5,
                      edgecolors='black', linewidth=2)

            ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
            ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
            ax.set_title(f'{method_name}\nXT Space', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')

        # Plot Score analysis (bottom row)
        for i, method in enumerate(self.methods):
            ax = axes[i, 1]
            traj = trajectories[method][0]  # Use first trajectory
            method_color = method_colors[method]
            method_name = method_names[method]

            score_pca_proj = score_pca.transform(traj['score'])
            n_points = len(score_pca_proj)
            colors = plt.cm.viridis(np.linspace(0, 1, n_points))

            # Plot trajectory
            for j in range(n_points - 1):
                ax.plot([score_pca_proj[j, 0], score_pca_proj[j+1, 0]],
                       [score_pca_proj[j, 1], score_pca_proj[j+1, 1]],
                       color=colors[j], linewidth=3, alpha=0.9)

            # Mark start and end points
            ax.scatter(score_pca_proj[0, 0], score_pca_proj[0, 1],
                      c=start_color, s=150, marker='o', label='Start', zorder=5,
                      edgecolors='black', linewidth=2)
            ax.scatter(score_pca_proj[-1, 0], score_pca_proj[-1, 1],
                      c=end_color, s=150, marker='s', label='End', zorder=5,
                      edgecolors='black', linewidth=2)

            ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
            ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
            ax.set_title(f'{method_name}\nScore Space', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')

        # Add row labels
        axes[0, 0].text(-0.15, 0.5, 'XT Analysis\n(Image Space)', transform=axes[0, 0].transAxes,
                        fontsize=14, fontweight='bold', rotation=90, va='center', ha='center')
        axes[1, 0].text(-0.15, 0.5, 'Score Analysis\n(Gradient Space)', transform=axes[1, 0].transAxes,
                        fontsize=14, fontweight='bold', rotation=90, va='center', ha='center')

        plt.tight_layout(rect=[0, 0, 1, 0.92])

        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'xt_vs_score_pca2_comparison_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\n✓ XT vs Score comparison plot saved to: {save_path}")

        plt.close()

    def generate_comparison_report(self, trajectories, xt_pca, score_pca, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """Generate comparison report between xt and score analysis"""
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(save_dir, f'xt_vs_score_analysis_report_{timestamp}.txt')

        with open(report_path, 'w') as f:
            f.write("XT vs Score PCA2 Analysis Comparison Report\n")
            f.write("="*60 + "\n\n")

            f.write("EXPERIMENTAL CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Methods: {', '.join(self.methods)}\n")
            f.write(f"Trajectories per method: {self.num_trajectories}\n")
            f.write(f"Inference steps per trajectory: {self.num_inference_steps}\n\n")

            f.write("PCA ANALYSIS RESULTS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"XT Analysis:\n")
            f.write(f"  - PC1 explained variance: {xt_pca.explained_variance_ratio_[0]:.4f}\n")
            f.write(f"  - PC2 explained variance: {xt_pca.explained_variance_ratio_[1]:.4f}\n")
            f.write(f"  - Total explained variance: {np.sum(xt_pca.explained_variance_ratio_):.4f}\n\n")

            f.write(f"Score Analysis:\n")
            f.write(f"  - PC1 explained variance: {score_pca.explained_variance_ratio_[0]:.4f}\n")
            f.write(f"  - PC2 explained variance: {score_pca.explained_variance_ratio_[1]:.4f}\n")
            f.write(f"  - Total explained variance: {np.sum(score_pca.explained_variance_ratio_):.4f}\n\n")

            f.write("ANALYSIS COMPARISON:\n")
            f.write("-" * 20 + "\n")
            f.write("XT Analysis (Image Space):\n")
            f.write("  - Represents actual image states during generation\n")
            f.write("  - Shows trajectory from noise to final image\n")
            f.write("  - More intuitive for visualization\n")
            f.write("  - Higher variance explained (image structure)\n\n")

            f.write("Score Analysis (Gradient Space):\n")
            f.write("  - Represents gradient/direction information\n")
            f.write("  - Shows how the model 'thinks' about generation\n")
            f.write("  - Better for zigzag analysis\n")
            f.write("  - May have lower variance explained (gradient noise)\n\n")

            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 15 + "\n")
            f.write("1. Use XT analysis for trajectory visualization\n")
            f.write("2. Use Score analysis for zigzag metrics\n")
            f.write("3. Combine both for comprehensive understanding\n")

        print(f"\n✓ Comparison report saved to: {report_path}")

        # Print summary to console
        print(f"\n" + "="*80)
        print("XT vs SCORE PCA2 ANALYSIS SUMMARY")
        print("="*80)

        print(f"\nXT Analysis (Image Space):")
        print(f"PC1 explained variance: {xt_pca.explained_variance_ratio_[0]:.4f}")
        print(f"PC2 explained variance: {xt_pca.explained_variance_ratio_[1]:.4f}")
        print(f"Total explained variance: {np.sum(xt_pca.explained_variance_ratio_):.4f}")

        print(f"\nScore Analysis (Gradient Space):")
        print(f"PC1 explained variance: {score_pca.explained_variance_ratio_[0]:.4f}")
        print(f"PC2 explained variance: {score_pca.explained_variance_ratio_[1]:.4f}")
        print(f"Total explained variance: {np.sum(score_pca.explained_variance_ratio_):.4f}")

def main():
    """Main function to run XT vs Score analysis"""

    print("🚀 XT vs Score PCA2 Analysis Comparison")
    print("="*60)
    print("Comparing trajectory analysis in image space vs gradient space")
    print("="*60)

    # Initialize analyzer
    analyzer = XTvsScoreAnalysis(n_samples=5000, num_inference_steps=25, num_trajectories=20)

    try:
        # Generate trajectories for all methods
        all_trajectories = {}

        for method in analyzer.methods:
            print(f"\n{'='*60}")
            print(f"Processing Method: {method.upper()}")
            print(f"{'='*60}")

            # Load pipeline
            pipe = analyzer.load_pipeline(method)

            # Generate trajectories
            trajectories = analyzer.generate_trajectories(pipe, method, seed=42)
            all_trajectories[method] = trajectories

            # Clean up GPU memory
            del pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Create PCA models for both XT and Score
        print(f"\n{'='*60}")
        print("Creating PCA models for XT and Score analysis...")
        print(f"{'='*60}")

        xt_pca = analyzer.create_pca_from_trajectories(all_trajectories, 'xt')
        score_pca = analyzer.create_pca_from_trajectories(all_trajectories, 'score')

        # Create comparison plots
        print(f"\n{'='*60}")
        print("Creating XT vs Score comparison plots...")
        print(f"{'='*60}")
        analyzer.plot_xt_vs_score_comparison(all_trajectories, xt_pca, score_pca)

        # Generate comparison report
        analyzer.generate_comparison_report(all_trajectories, xt_pca, score_pca)

        print(f"\n✅ XT vs Score analysis completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
