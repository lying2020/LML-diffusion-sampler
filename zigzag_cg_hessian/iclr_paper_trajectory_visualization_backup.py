#!/usr/bin/env python3
"""
ICLR Paper Format: DDPM vs Hessian-Free Trajectory Evolution Visualization
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

# Import schedulers
from diffusers import DDPMPipeline, DDPMScheduler
from scheduler.scheduling_dpmsolver_multistep_hcg import DPMSolverMultistepHCGScheduler
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

class ICLRTrajectoryVisualizer:
    """ICLR paper format trajectory visualizer for DDPM vs Hessian-Free"""

    def __init__(self, n_samples=5000, num_inference_steps=25, num_trajectories=100):
        self.n_samples = n_samples
        self.num_inference_steps = num_inference_steps
        self.num_trajectories = num_trajectories

        print(f"🔧 ICLR Paper Format Configuration:")
        print(f"   - CIFAR-10 samples for PCA: {self.n_samples}")
        print(f"   - Inference steps per trajectory: {self.num_inference_steps}")
        print(f"   - Number of trajectories per method: {self.num_trajectories}")
        print(f"   - Total sampling steps: {self.num_trajectories * self.num_inference_steps}")

    def load_pipeline(self, method_name):
        """Load pipeline for different sampling methods"""
        model_path = os.path.join(project.model_dir, 'ddpm_ema_cifar10')

        print(f"\n🔧 Loading {method_name} pipeline...")
        pipe = DDPMPipeline.from_pretrained(model_path, torch_dtype=torch.float32, use_safetensors=False)
        pipe.unet.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup scheduler based on method
        if method_name == 'DDPM':
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.set_timesteps(self.num_inference_steps)
        elif method_name == 'Hessian_Free':
            pipe.scheduler = DPMSolverMultistepHCGScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lamb = 0.0008
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = 1e-8
            pipe.scheduler.hessian_method = 'hessian_free'
            pipe.scheduler.set_model(pipe.unet)
            pipe.scheduler.set_timesteps(self.num_inference_steps)

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

            # DDPM/Hessian-Free step
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

    def plot_iclr_trajectory_comparison(self, trajectories, pca_model, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """Plot ICLR paper format trajectory comparison - 4 subplots in one row"""
        os.makedirs(save_dir, exist_ok=True)

        # Project trajectories to PCA space
        ddpm_traj = trajectories['DDPM'][0]  # Use first trajectory for visualization
        hessian_traj = trajectories['Hessian_Free'][0]

        ddpm_pca = pca_model.transform(ddpm_traj['xt'])
        hessian_pca = pca_model.transform(hessian_traj['xt'])

        # Create figure with 4 subplots in one row - optimized for ICLR paper format
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # Single row, 4 columns
        fig.suptitle('Trajectory Evolution: DDPM vs Hessian-Free Methods',
                     fontsize=18, fontweight='bold', y=0.95)

        # Define colors for ICLR paper format
        ddpm_color = '#E74C3C'  # Red
        hessian_color = '#3498DB'  # Blue
        start_color = '#27AE60'  # Green
        end_color = '#E67E22'  # Orange

        # Plot 1: DDPM Trajectory Evolution
        ax1 = axes[0]
        self._plot_single_trajectory_iclr(ax1, ddpm_pca, 'DDPM', ddpm_color, start_color, end_color)

        # Plot 2: Hessian-Free Trajectory Evolution
        ax2 = axes[1]
        self._plot_single_trajectory_iclr(ax2, hessian_pca, 'Hessian-Free', hessian_color, start_color, end_color)

        # Plot 3: Side-by-side comparison
        ax3 = axes[2]
        self._plot_comparison_trajectories_iclr(ax3, ddpm_pca, hessian_pca, ddpm_color, hessian_color, start_color, end_color)

        # Plot 4: Step-by-step evolution
        ax4 = axes[3]
        self._plot_step_evolution_iclr(ax4, ddpm_pca, hessian_pca, ddpm_color, hessian_color)

        # Adjust layout for ICLR paper format
        plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space for main title

        # Save the plot with high DPI for ICLR paper
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'iclr_trajectory_evolution_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\n✓ ICLR format trajectory evolution plot saved to: {save_path}")

        plt.show()

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

    def _plot_comparison_trajectories_iclr(self, ax, ddpm_pca, hessian_pca, ddpm_color, hessian_color, start_color, end_color):
        """Plot side-by-side trajectory comparison optimized for ICLR paper format"""
        # DDPM trajectory with thick line
        ax.plot(ddpm_pca[:, 0], ddpm_pca[:, 1], color=ddpm_color, linewidth=4, alpha=0.9, label='DDPM Path')
        ax.scatter(ddpm_pca[0, 0], ddpm_pca[0, 1], c=start_color, s=150, marker='o', zorder=5)
        ax.scatter(ddpm_pca[-1, 0], ddpm_pca[-1, 1], c=end_color, s=150, marker='s', zorder=5)

        # Hessian-Free trajectory with thick line
        ax.plot(hessian_pca[:, 0], hessian_pca[:, 1], color=hessian_color, linewidth=4, alpha=0.9, label='Hessian-Free Path')
        ax.scatter(hessian_pca[0, 0], hessian_pca[0, 1], c=start_color, s=150, marker='o', zorder=5)
        ax.scatter(hessian_pca[-1, 0], hessian_pca[-1, 1], c=end_color, s=150, marker='s', zorder=5)

        ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax.set_title('Trajectory Comparison\n(Red: DDPM, Blue: Hessian-Free)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.axis('equal')

        # Make axes labels bold
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)

    def _plot_step_evolution_iclr(self, ax, ddpm_pca, hessian_pca, ddpm_color, hessian_color):
        """Plot step-by-step evolution showing convergence optimized for ICLR paper format"""
        n_steps = len(ddpm_pca)

        # Calculate distance from end point for each step
        ddpm_distances = []
        hessian_distances = []

        for i in range(n_steps):
            ddpm_dist = np.linalg.norm(ddpm_pca[i] - ddpm_pca[-1])
            hessian_dist = np.linalg.norm(hessian_pca[i] - hessian_pca[-1])
            ddpm_distances.append(ddpm_dist)
            hessian_distances.append(hessian_dist)

        # Plot convergence curves with thick lines
        steps = range(n_steps)
        ax.plot(steps, ddpm_distances, color=ddpm_color, linewidth=4, alpha=0.9, label='DDPM Convergence')
        ax.plot(steps, hessian_distances, color=hessian_color, linewidth=4, alpha=0.9, label='Hessian-Free Convergence')

        ax.set_xlabel('Sampling Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance to Final Point', fontsize=12, fontweight='bold')
        ax.set_title('Convergence Analysis\n(Distance to End Point)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.set_yscale('log')

        # Make axes labels bold
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)

def main():
    """Main function to run ICLR paper format trajectory visualization"""

    print("🚀 ICLR Paper Format: DDPM vs Hessian-Free Trajectory Visualization")
    print("="*70)
    print("Optimized for single-column paper format with thicker lines and larger fonts")
    print("="*70)

    # Initialize visualizer
    visualizer = ICLRTrajectoryVisualizer(n_samples=5000, num_inference_steps=25, num_trajectories=100)

    try:
        # Generate trajectories for both methods
        methods = ['DDPM', 'Hessian_Free']
        all_trajectories = {}

        for method in methods:
            print(f"\n{'='*60}")
            print(f"Processing Method: {method}")
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

        # Create ICLR paper format trajectory comparison plots
        print(f"\n{'='*60}")
        print("Creating ICLR Paper Format Trajectory Visualization...")
        print(f"{'='*60}")
        visualizer.plot_iclr_trajectory_comparison(all_trajectories, pca_model)

        print(f"\n✅ ICLR paper format trajectory visualization completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
