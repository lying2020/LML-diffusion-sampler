#!/usr/bin/env python3
"""
ICLR Paper Format: DDIM vs Hessian-Free Anisotropy Visualization
Using max and median eigenvalues to show isotropic to anisotropic transition
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
from diffusers import DDPMPipeline, DDPMScheduler, DDIMScheduler
from scheduler.scheduling_dpmsolver_hessian_free import DPMSolverMultistepHessianFreeScheduler
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

class ICLRAnisotropyVisualizer:
    """ICLR paper format visualizer for DDIM vs Hessian-Free anisotropy analysis"""

    def __init__(self, n_samples=5000, num_inference_steps=25, num_trajectories=100):
        self.n_samples = n_samples
        self.num_inference_steps = num_inference_steps
        self.num_trajectories = num_trajectories

        print(f"🔧 ICLR Anisotropy Configuration:")
        print(f"   - CIFAR-10 samples for PCA: {self.n_samples}")
        print(f"   - Inference steps per trajectory: {self.num_inference_steps}")
        print(f"   - Number of trajectories per method: {self.num_trajectories}")
        print(f"   - Total sampling steps: {self.num_trajectories * self.num_inference_steps}")
        print(f"   - PCA method: Max and Median eigenvalues (anisotropy analysis)")

    def load_pipeline(self, method_name):
        """Load pipeline for different sampling methods"""
        model_path = os.path.join(project.model_dir, 'ddpm_ema_cifar10')

        print(f"\n🔧 Loading {method_name} pipeline...")
        pipe = DDPMPipeline.from_pretrained(model_path, torch_dtype=torch.float32, use_safetensors=False)
        pipe.unet.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup scheduler based on method
        if method_name == 'DDIM':
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.set_timesteps(self.num_inference_steps)
        elif method_name == 'Hessian_Free':
            pipe.scheduler = DPMSolverMultistepHessianFreeScheduler.from_config(pipe.scheduler.config)
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

            # DDIM/Hessian-Free step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Convert to numpy arrays
        for key in ['xt', 'score', 'timesteps', 'noise_pred']:
            trajectory_data[key] = np.array(trajectory_data[key])

        return trajectory_data

    def create_anisotropy_pca(self, trajectories):
        """Create PCA model using max and median eigenvalues for anisotropy analysis"""
        print(f"\n📊 Creating PCA model for anisotropy analysis...")

        # Collect all trajectory data
        all_trajectory_data = []
        for method, traj_list in trajectories.items():
            for traj in traj_list:
                all_trajectory_data.append(traj['xt'])

        # Stack all trajectories
        all_data = np.vstack(all_trajectory_data)
        print(f"  Total trajectory data shape: {all_data.shape}")

        # Perform full PCA to get all eigenvalues
        print(f"  Computing full PCA to extract eigenvalues...")
        pca_full = PCA(n_components=min(all_data.shape[0], all_data.shape[1]), svd_solver='full')
        pca_full.fit(all_data)

        # Get eigenvalues (explained variance)
        eigenvalues = pca_full.explained_variance_
        print(f"  Total eigenvalues computed: {len(eigenvalues)}")
        print(f"  Max eigenvalue: {eigenvalues[0]:.6f}")
        print(f"  Median eigenvalue: {eigenvalues[len(eigenvalues)//2]:.6f}")
        print(f"  Min eigenvalue: {eigenvalues[-1]:.6f}")

        # Calculate anisotropy metrics
        max_eigenvalue = eigenvalues[0]
        median_eigenvalue = eigenvalues[len(eigenvalues)//2]
        min_eigenvalue = eigenvalues[-1]

        anisotropy_ratio_max_median = max_eigenvalue / median_eigenvalue
        anisotropy_ratio_max_min = max_eigenvalue / min_eigenvalue

        print(f"  Anisotropy ratio (max/median): {anisotropy_ratio_max_median:.2f}")
        print(f"  Anisotropy ratio (max/min): {anisotropy_ratio_max_min:.2f}")

        # Create custom PCA with max and median eigenvalues
        print(f"  Creating custom PCA with max and median eigenvalues...")

        # Get the principal components corresponding to max and median eigenvalues
        max_pc = pca_full.components_[0]  # First principal component (max eigenvalue)
        median_idx = len(eigenvalues) // 2
        median_pc = pca_full.components_[median_idx]  # Median principal component

        # Create custom PCA model
        class AnisotropyPCA:
            def __init__(self, max_pc, median_pc, max_eigenvalue, median_eigenvalue):
                self.components_ = np.vstack([max_pc, median_pc])
                self.explained_variance_ = np.array([max_eigenvalue, median_eigenvalue])
                self.explained_variance_ratio_ = self.explained_variance_ / np.sum(pca_full.explained_variance_)
                self.mean_ = pca_full.mean_
                self.anisotropy_ratio = max_eigenvalue / median_eigenvalue

            def transform(self, X):
                # Center the data
                X_centered = X - self.mean_
                # Project onto max and median eigenvectors
                return X_centered @ self.components_.T

        pca_model = AnisotropyPCA(
            max_pc, median_pc,
            max_eigenvalue, median_eigenvalue
        )

        print(f"✓ Anisotropy PCA completed.")
        print(f"  Max eigenvalue ratio: {pca_model.explained_variance_ratio_[0]:.4f}")
        print(f"  Median eigenvalue ratio: {pca_model.explained_variance_ratio_[1]:.4f}")
        print(f"  Anisotropy ratio: {pca_model.anisotropy_ratio:.2f}")

        return pca_model, eigenvalues

    def plot_iclr_anisotropy_comparison(self, trajectories, pca_model, eigenvalues, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """Plot ICLR paper format anisotropy comparison - 4 subplots in one row"""
        os.makedirs(save_dir, exist_ok=True)

        # Project trajectories to PCA space
        ddpm_traj = trajectories['DDIM'][0]  # Use first trajectory for visualization
        hessian_traj = trajectories['Hessian_Free'][0]

        ddpm_pca = pca_model.transform(ddpm_traj['xt'])
        hessian_pca = pca_model.transform(hessian_traj['xt'])

        # Calculate anisotropy ratios for both methods
        ddpm_anisotropy = []
        hessian_anisotropy = []

        for i in range(len(ddpm_pca)):
            if i > 0:
                ddpm_ratio = np.abs(ddpm_pca[i, 0]) / (np.abs(ddpm_pca[i, 1]) + 1e-8)
                hessian_ratio = np.abs(hessian_pca[i, 0]) / (np.abs(hessian_pca[i, 1]) + 1e-8)
                ddpm_anisotropy.append(ddpm_ratio)
                hessian_anisotropy.append(hessian_ratio)

        # Create figure with 4 subplots in one row - optimized for ICLR paper format
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # Single row, 4 columns
        fig.suptitle('DDIM vs Hessian-Free: Anisotropy Analysis\n(Max vs Median Eigenvalues - Isotropic to Anisotropic Transition)',
                     fontsize=18, fontweight='bold', y=0.95)

        # Define colors for ICLR paper format
        ddpm_color = '#E74C3C'  # Red
        hessian_color = '#3498DB'  # Blue
        start_color = '#27AE60'  # Green
        end_color = '#E67E22'  # Orange

        # Plot 1: DDIM Trajectory Evolution (Anisotropy)
        ax1 = axes[0]
        self._plot_single_trajectory_iclr_anisotropy(ax1, ddpm_pca, 'DDIM', ddpm_color, start_color, end_color, pca_model)

        # Plot 2: Hessian-Free Trajectory Evolution (Anisotropy)
        ax2 = axes[1]
        self._plot_single_trajectory_iclr_anisotropy(ax2, hessian_pca, 'Hessian-Free', hessian_color, start_color, end_color, pca_model)

        # Plot 3: Side-by-side comparison (Anisotropy)
        ax3 = axes[2]
        self._plot_comparison_trajectories_iclr_anisotropy(ax3, ddpm_pca, hessian_pca, ddpm_color, hessian_color, start_color, end_color, pca_model)

        # Plot 4: Anisotropy evolution over time
        ax4 = axes[3]
        self._plot_anisotropy_evolution_iclr(ax4, ddpm_anisotropy, hessian_anisotropy, ddpm_color, hessian_color)

        # Adjust layout for ICLR paper format
        plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space for main title

        # Save the plot with high DPI for ICLR paper
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'iclr_anisotropy_analysis_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\n✓ ICLR format anisotropy analysis plot saved to: {save_path}")

        plt.show()

    def _plot_single_trajectory_iclr_anisotropy(self, ax, trajectory_pca, method_name, method_color, start_color, end_color, pca_model):
        """Plot single trajectory evolution optimized for ICLR paper format with anisotropy"""
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

        ax.set_xlabel('Max Eigenvalue PC', fontsize=12, fontweight='bold')
        ax.set_ylabel('Median Eigenvalue PC', fontsize=12, fontweight='bold')
        ax.set_title(f'{method_name}\nAnisotropy Trajectory', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.axis('equal')

        # Make axes labels bold
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)

    def _plot_comparison_trajectories_iclr_anisotropy(self, ax, ddpm_pca, hessian_pca, ddpm_color, hessian_color, start_color, end_color, pca_model):
        """Plot side-by-side trajectory comparison optimized for ICLR paper format with anisotropy"""
        # DDPM trajectory with thick line
        ax.plot(ddpm_pca[:, 0], ddpm_pca[:, 1], color=ddpm_color, linewidth=4, alpha=0.9, label='DDIM Path')
        ax.scatter(ddpm_pca[0, 0], ddpm_pca[0, 1], c=start_color, s=150, marker='o', zorder=5)
        ax.scatter(ddpm_pca[-1, 0], ddpm_pca[-1, 1], c=end_color, s=150, marker='s', zorder=5)

        # Hessian-Free trajectory with thick line
        ax.plot(hessian_pca[:, 0], hessian_pca[:, 1], color=hessian_color, linewidth=4, alpha=0.9, label='Hessian-Free Path')
        ax.scatter(hessian_pca[0, 0], hessian_pca[0, 1], c=start_color, s=150, marker='o', zorder=5)
        ax.scatter(hessian_pca[-1, 0], hessian_pca[-1, 1], c=end_color, s=150, marker='s', zorder=5)

        ax.set_xlabel('Max Eigenvalue PC', fontsize=12, fontweight='bold')
        ax.set_ylabel('Median Eigenvalue PC', fontsize=12, fontweight='bold')
        ax.set_title('Anisotropy Comparison\n(Red: DDIM, Blue: Hessian-Free)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.axis('equal')

        # Make axes labels bold
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)

    def _plot_anisotropy_evolution_iclr(self, ax, ddpm_anisotropy, hessian_anisotropy, ddpm_color, hessian_color):
        """Plot anisotropy evolution over time optimized for ICLR paper format"""
        # Plot anisotropy evolution curves with thick lines
        steps = range(len(ddpm_anisotropy))
        ax.plot(steps, ddpm_anisotropy, color=ddpm_color, linewidth=4, alpha=0.9, label='DDIM Anisotropy')
        ax.plot(steps, hessian_anisotropy, color=hessian_color, linewidth=4, alpha=0.9, label='Hessian-Free Anisotropy')

        # Add horizontal line for isotropic reference
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Isotropic (1.0)')

        ax.set_xlabel('Sampling Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Anisotropy Ratio (Max/Median)', fontsize=12, fontweight='bold')
        ax.set_title('Anisotropy Evolution\n(Isotropic → Anisotropic)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.set_yscale('log')

        # Make axes labels bold
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)

def main():
    """Main function to run ICLR paper format anisotropy visualization"""

    print("🚀 ICLR Paper Format: DDIM vs Hessian-Free Anisotropy Visualization")
    print("="*80)
    print("Using max and median eigenvalues to show isotropic to anisotropic transition")
    print("="*80)

    # Initialize visualizer
    visualizer = ICLRAnisotropyVisualizer(n_samples=5000, num_inference_steps=25, num_trajectories=100)

    try:
        # Generate trajectories for both methods
        methods = ['DDIM', 'Hessian_Free']
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

        # Create anisotropy PCA model from all trajectory data
        print(f"\n{'='*60}")
        print("Creating anisotropy PCA model...")
        print(f"{'='*60}")
        pca_model, eigenvalues = visualizer.create_anisotropy_pca(all_trajectories)

        # Create ICLR paper format anisotropy comparison plots
        print(f"\n{'='*60}")
        print("Creating ICLR Paper Format Anisotropy Visualization...")
        print(f"{'='*60}")
        visualizer.plot_iclr_anisotropy_comparison(all_trajectories, pca_model, eigenvalues)

        print(f"\n✅ ICLR paper format anisotropy visualization completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
