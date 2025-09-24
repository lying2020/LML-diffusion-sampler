#!/usr/bin/env python3
"""
ICLR Paper Format: Multi-Method Trajectory Evolution Visualization
Support for ["pndm", "ddim", "dpm++", "dpm", "unipc"] methods
Optimized for single-column paper format with thicker lines and larger fonts
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
import torch
import sys
import os
import time
from datetime import datetime

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
    'lines.linewidth': 3,
    'lines.markersize': 8,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3
})

class ICLRMultiMethodTrajectoryVisualizer:
    """ICLR paper format trajectory visualizer for multiple sampling methods"""

    def __init__(self, n_samples=5000, num_inference_steps=25, num_trajectories=100):
        self.n_samples = n_samples
        self.num_inference_steps = num_inference_steps
        self.num_trajectories = num_trajectories

        # 支持的方法列表
        self.supported_methods = ["pndm", "ddim", "dpm++", "dpm", "unipc"]

        # 为每个方法定义颜色
        self.method_colors = {
            'pndm': '#E74C3C',      # Red
            'ddim': '#3498DB',      # Blue
            'dpm++': '#9B59B6',     # Purple
            'dpm': '#E67E22',       # Orange
            'unipc': '#2ECC71'      # Green
        }

        print(f"🔧 ICLR Paper Format Configuration:")
        print(f"   - CIFAR-10 samples for PCA: {self.n_samples}")
        print(f"   - Inference steps per trajectory: {self.num_inference_steps}")
        print(f"   - Number of trajectories per method: {self.num_trajectories}")
        print(f"   - Supported methods: {', '.join(self.supported_methods)}")
        print(f"   - Total sampling steps: {self.num_trajectories * self.num_inference_steps}")

    def load_pipeline(self, method_name):
        """Load pipeline for different sampling methods - 参考cifar10.py的方式"""
        model_id = os.path.join(project.model_dir, 'ddpm_ema_cifar10')

        print(f"\n🔧 Loading {method_name} pipeline...")
        pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        pipe.unet.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup scheduler based on method - 参考cifar10.py的scheduler设置
        if method_name == 'pndm':
            pipe.scheduler = PNDMSchedulerLM.from_config(pipe.scheduler.config)
        elif method_name == 'ddim':
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif method_name == 'dpm++':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver++"
            pipe.scheduler.lm = False
        elif method_name == 'dpm':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lm = False
        elif method_name == 'unipc':
            pipe.scheduler = UniPCMultistepSchedulerLM.from_config(pipe.scheduler.config)
        else:
            raise ValueError(f"Unsupported method: {method_name}")

        print(f"✓ {method_name} pipeline loaded successfully")
        return pipe

    def generate_trajectories(self, pipe, method_name, seed=42):
        """Generate multiple trajectories for statistical analysis"""
        print(f"\n🚀 Generating {self.num_trajectories} trajectories using {method_name}...")

        trajectories = []
        torch.manual_seed(seed)

        for i in range(self.num_trajectories):
            if i % 20 == 0:
                print(f"  Generating trajectory {i+1}/{self.num_trajectories}")

            # Generate single trajectory
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
            'xt': [],  # State vectors
            'score': [],  # Score/drift vectors
            'timesteps': [],
            'noise_pred': []
        }

        # Get scheduler
        scheduler = pipe.scheduler

        scheduler.set_timesteps(self.num_inference_steps)
        for j, t in enumerate(scheduler.timesteps):
            # Store current state (detach to avoid grad issues)
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

    def create_pca_from_trajectories(self, trajectories):
        """Create PCA model from trajectory data to ensure dimension compatibility"""
        print(f"\n📊 Creating PCA model from trajectory data...")

        # Collect all trajectory data
        all_trajectory_data = []
        for method, traj_list in trajectories.items():
            for traj in traj_list:
                all_trajectory_data.append(traj['xt'])

        # Stack all trajectories
        all_data = np.vstack(all_trajectory_data)
        print(f"  Total trajectory data shape: {all_data.shape}")

        # Perform PCA
        pca_model = PCA(n_components=2, svd_solver='randomized')
        pca_model.fit(all_data)

        print(f"✓ PCA completed. Explained variance: {pca_model.explained_variance_ratio_}")
        return pca_model

    def plot_iclr_trajectory_evolution(self, trajectories, pca_model, save_dir='./zigzag_cg_hessian'):
        """Plot ICLR paper format trajectory evolution - 5 methods in one row"""
        os.makedirs(save_dir, exist_ok=True)

        # Create figure with 5 subplots in one row - optimized for ICLR paper format
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # Single row, 5 columns
        fig.suptitle('Trajectory Evolution: Multi-Method Comparison',
                     fontsize=18, fontweight='bold', y=0.95)

        # Define colors
        start_color = '#27AE60'  # Green
        end_color = '#E67E22'    # Orange

        # Plot each method
        for i, method in enumerate(self.supported_methods):
            if method in trajectories and len(trajectories[method]) > 0:
                ax = axes[i]
                trajectory_pca = pca_model.transform(trajectories[method][0]['xt'])
                method_color = self.method_colors[method]

                self._plot_single_trajectory_iclr(ax, trajectory_pca, method.upper(),
                                                method_color, start_color, end_color)

        # Adjust layout for ICLR paper format
        plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space for main title

        # Save the plot with high DPI for ICLR paper
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'iclr_trajectory_evolution_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\n✓ ICLR format trajectory evolution plot saved to: {save_path}")

        plt.close()

    def plot_iclr_convergence_analysis(self, trajectories, pca_model, save_dir='./zigzag_cg_hessian'):
        """Plot ICLR paper format convergence analysis - 5 methods in one row"""
        os.makedirs(save_dir, exist_ok=True)

        # Create figure with 5 subplots in one row
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle('Convergence Analysis: Multi-Method Comparison',
                     fontsize=18, fontweight='bold', y=0.95)

        # Plot each method
        for i, method in enumerate(self.supported_methods):
            if method in trajectories and len(trajectories[method]) > 0:
                ax = axes[i]
                trajectory_pca = pca_model.transform(trajectories[method][0]['xt'])
                method_color = self.method_colors[method]

                self._plot_convergence_single_method(ax, trajectory_pca, method.upper(), method_color)

        # Adjust layout for ICLR paper format
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        # Save the plot with high DPI for ICLR paper
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'iclr_convergence_analysis_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\n✓ ICLR format convergence analysis plot saved to: {save_path}")

        plt.close()

    def _plot_single_trajectory_iclr(self, ax, trajectory_pca, method_name, method_color, start_color, end_color):
        """Plot single trajectory evolution optimized for ICLR paper format"""
        # Create color gradient from start to end
        n_points = len(trajectory_pca)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))

        # Plot trajectory with thick lines
        for i in range(n_points - 1):
            ax.plot([trajectory_pca[i, 0], trajectory_pca[i+1, 0]],
                   [trajectory_pca[i, 1], trajectory_pca[i+1, 1]],
                   color=colors[i], linewidth=4, alpha=0.9)

        # Mark start and end points with larger markers
        ax.scatter(trajectory_pca[0, 0], trajectory_pca[0, 1],
                  c=start_color, s=200, marker='o', label='Start', zorder=5,
                  edgecolors='black', linewidth=2)
        ax.scatter(trajectory_pca[-1, 0], trajectory_pca[-1, 1],
                  c=end_color, s=200, marker='s', label='End', zorder=5,
                  edgecolors='black', linewidth=2)

        # Add step markers with larger size
        for i in range(0, n_points, max(1, n_points//8)):
            ax.scatter(trajectory_pca[i, 0], trajectory_pca[i, 1],
                      c=colors[i], s=80, marker='o', alpha=0.8, zorder=3)

        ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax.set_title(f'{method_name}\nTrajectory Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.axis('equal')

        # Make axes labels bold
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)

    def _plot_convergence_single_method(self, ax, trajectory_pca, method_name, method_color):
        """Plot convergence analysis for single method optimized for ICLR paper format"""
        n_steps = len(trajectory_pca)

        # Calculate distance from end point for each step
        distances = []
        for i in range(n_steps):
            dist = np.linalg.norm(trajectory_pca[i] - trajectory_pca[-1])
            distances.append(dist)

        # Plot convergence curve with thick line
        steps = range(n_steps)
        ax.plot(steps, distances, color=method_color, linewidth=4, alpha=0.9,
                label=f'{method_name} Convergence')

        ax.set_xlabel('Sampling Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance to Final Point', fontsize=12, fontweight='bold')
        ax.set_title(f'{method_name}\nConvergence Analysis', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.set_yscale('log')

        # Make axes labels bold
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)

    def plot_iclr_comprehensive_comparison(self, trajectories, pca_model, save_dir='./zigzag_cg_hessian'):
        """Plot comprehensive comparison with all methods in one figure"""
        os.makedirs(save_dir, exist_ok=True)

        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Multi-Method Analysis', fontsize=20, fontweight='bold', y=0.95)

        # Plot 1: All methods trajectory comparison (top-left, spans 2 columns)
        ax1 = axes[0, 0]
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
        self._plot_all_methods_comparison(ax1, trajectories, pca_model)

        # Plot 2: Convergence comparison (top-right)
        ax2 = axes[0, 2]
        self._plot_convergence_comparison(ax2, trajectories, pca_model)

        # Plot 3-7: Individual method trajectories (bottom row)
        for i, method in enumerate(self.supported_methods[:3]):
            if method in trajectories and len(trajectories[method]) > 0:
                ax = axes[1, i]
                trajectory_pca = pca_model.transform(trajectories[method][0]['xt'])
                method_color = self.method_colors[method]

                self._plot_single_trajectory_iclr(ax, trajectory_pca, method.upper(),
                                                method_color, '#27AE60', '#E67E22')

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'iclr_comprehensive_comparison_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\n✓ ICLR format comprehensive comparison plot saved to: {save_path}")

        plt.close()

    def _plot_all_methods_comparison(self, ax, trajectories, pca_model):
        """Plot all methods in one subplot for comparison"""
        for method in self.supported_methods:
            if method in trajectories and len(trajectories[method]) > 0:
                trajectory_pca = pca_model.transform(trajectories[method][0]['xt'])
                method_color = self.method_colors[method]

                ax.plot(trajectory_pca[:, 0], trajectory_pca[:, 1],
                       color=method_color, linewidth=3, alpha=0.8, label=method.upper())

                # Mark start and end points
                ax.scatter(trajectory_pca[0, 0], trajectory_pca[0, 1],
                          c=method_color, s=100, marker='o', alpha=0.8)
                ax.scatter(trajectory_pca[-1, 0], trajectory_pca[-1, 1],
                          c=method_color, s=100, marker='s', alpha=0.8)

        ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax.set_title('All Methods Trajectory Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.axis('equal')

    def _plot_convergence_comparison(self, ax, trajectories, pca_model):
        """Plot convergence comparison for all methods"""
        for method in self.supported_methods:
            if method in trajectories and len(trajectories[method]) > 0:
                trajectory_pca = pca_model.transform(trajectories[method][0]['xt'])
                method_color = self.method_colors[method]

                n_steps = len(trajectory_pca)
                distances = []
                for i in range(n_steps):
                    dist = np.linalg.norm(trajectory_pca[i] - trajectory_pca[-1])
                    distances.append(dist)

                steps = range(n_steps)
                ax.plot(steps, distances, color=method_color, linewidth=3, alpha=0.8,
                       label=method.upper())

        ax.set_xlabel('Sampling Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance to Final Point', fontsize=12, fontweight='bold')
        ax.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.set_yscale('log')

def main():
    """Main function to run ICLR paper format multi-method trajectory visualization"""

    print("🚀 ICLR Paper Format: Multi-Method Trajectory Visualization")
    print("="*70)
    print("Support for PNDM, DDIM, DPM++, DPM, UniPC methods")
    print("Optimized for single-column paper format with thicker lines and larger fonts")
    print("="*70)

    # Initialize visualizer
    visualizer = ICLRMultiMethodTrajectoryVisualizer(n_samples=5000, num_inference_steps=25, num_trajectories=100)

    try:
        # Generate trajectories for all methods
        all_trajectories = {}

        for method in visualizer.supported_methods:
            print(f"\n{'='*60}")
            print(f"Processing Method: {method.upper()}")
            print(f"{'='*60}")

            # Load pipeline
            pipe = visualizer.load_pipeline(method)

            # Generate trajectories
            trajectories = visualizer.generate_trajectories(pipe, method, seed=42)
            all_trajectories[method] = trajectories

            # Clean up GPU memory
            del pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Create PCA model from all trajectory data
        print(f"\n{'='*60}")
        print("Creating PCA model from all trajectory data...")
        print(f"{'='*60}")
        pca_model = visualizer.create_pca_from_trajectories(all_trajectories)

        # Create ICLR paper format visualizations
        print(f"\n{'='*60}")
        print("Creating ICLR Paper Format Visualizations...")
        print(f"{'='*60}")

        # 1. Trajectory Evolution (5 methods in one row)
        visualizer.plot_iclr_trajectory_evolution(all_trajectories, pca_model)

        # 2. Convergence Analysis (5 methods in one row)
        visualizer.plot_iclr_convergence_analysis(all_trajectories, pca_model)

        # 3. Comprehensive Comparison (2x3 layout)
        visualizer.plot_iclr_comprehensive_comparison(all_trajectories, pca_model)

        print(f"\n✅ ICLR paper format multi-method visualization completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
