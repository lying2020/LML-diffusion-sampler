#!/usr/bin/env python3
"""
Trajectory Evolution Visualization
Shows how the sampling path evolves from start to end for DDPM vs Hessian-Free
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

class TrajectoryEvolutionVisualizer:
    """Visualizer for trajectory evolution from start to end"""

    def __init__(self, n_samples=5000, num_inference_steps=25):
        self.n_samples = n_samples
        self.num_inference_steps = num_inference_steps

    def load_cifar10_samples(self):
        """Load CIFAR-10 samples for PCA reference"""
        print(f"Loading {self.n_samples} CIFAR-10 samples...")
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        imgs = []
        for i in range(min(self.n_samples, len(dataset))):
            img, _ = dataset[i]
            # Convert to grayscale and flatten
            gray = 0.2989*img[0].numpy() + 0.5870*img[1].numpy() + 0.1140*img[2].numpy()
            imgs.append(gray.ravel())

        X = np.stack(imgs, axis=0).astype(np.float64)
        X -= X.mean(axis=0, keepdims=True)
        print(f"✓ Loaded {X.shape[0]} CIFAR-10 samples")
        return X

    def load_pipeline(self, method_name):
        """Load pipeline for different sampling methods"""
        model_path = os.path.join(project.model_dir, 'ddpm_ema_cifar10')

        print(f"Loading {method_name} pipeline...")
        pipe = DDPMPipeline.from_pretrained(model_path, torch_dtype=torch.float32, use_safetensors=False)
        pipe.unet.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup scheduler based on method
        if method_name == 'DDPM':
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        elif method_name == 'Hessian_Free':
            pipe.scheduler = DPMSolverMultistepHCGScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lamb = 0.0008
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = 1e-8
            pipe.scheduler.hessian_method = 'hessian_free'
            pipe.scheduler.set_model(pipe.unet)

        print(f"✓ {method_name} pipeline loaded successfully")
        return pipe

    def generate_single_trajectory(self, pipe, method_name, seed=42):
        """Generate a single trajectory for visualization"""
        torch.manual_seed(seed)

        # Initialize random noise
        device = pipe.unet.device
        shape = (1, pipe.unet.config.in_channels, 32, 32)
        latents = torch.randn(shape, device=device)

        # Store trajectory data
        trajectory_data = {
            'xt': [],  # State vectors
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

            # DDPM/Hessian-Free step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Convert to numpy arrays
        for key in ['xt', 'timesteps', 'noise_pred']:
            trajectory_data[key] = np.array(trajectory_data[key])

        return trajectory_data

    def plot_trajectory_evolution(self, ddpm_traj, hessian_traj, pca_model, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """Plot trajectory evolution showing path from start to end"""
        os.makedirs(save_dir, exist_ok=True)

        # Project trajectories to PCA space
        ddpm_pca = pca_model.transform(ddpm_traj['xt'])
        hessian_pca = pca_model.transform(hessian_traj['xt'])

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Trajectory Evolution: From Start to End', fontsize=16, fontweight='bold')

        # Plot 1: DDPM Trajectory Evolution
        ax1 = axes[0, 0]
        self._plot_single_trajectory_evolution(ax1, ddpm_pca, 'DDPM', 'red')

        # Plot 2: Hessian-Free Trajectory Evolution
        ax2 = axes[0, 1]
        self._plot_single_trajectory_evolution(ax2, hessian_pca, 'Hessian-Free', 'blue')

        # Plot 3: Side-by-side comparison
        ax3 = axes[1, 0]
        self._plot_comparison_trajectories(ax3, ddpm_pca, hessian_pca)

        # Plot 4: Step-by-step evolution
        ax4 = axes[1, 1]
        self._plot_step_evolution(ax4, ddpm_pca, hessian_pca)

        plt.tight_layout()

        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'trajectory_evolution_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Trajectory evolution plot saved to: {save_path}")

        plt.show()

    def _plot_single_trajectory_evolution(self, ax, trajectory_pca, method_name, color):
        """Plot single trajectory evolution with color gradient"""
        # Create color gradient from start to end
        n_points = len(trajectory_pca)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))

        # Plot trajectory with color gradient
        for i in range(n_points - 1):
            ax.plot([trajectory_pca[i, 0], trajectory_pca[i+1, 0]],
                   [trajectory_pca[i, 1], trajectory_pca[i+1, 1]],
                   color=colors[i], linewidth=3, alpha=0.8)

        # Mark start and end points
        ax.scatter(trajectory_pca[0, 0], trajectory_pca[0, 1],
                  c='green', s=200, marker='o', label='Start', zorder=5, edgecolors='black', linewidth=2)
        ax.scatter(trajectory_pca[-1, 0], trajectory_pca[-1, 1],
                  c='red', s=200, marker='s', label='End', zorder=5, edgecolors='black', linewidth=2)

        # Add step markers
        for i in range(0, n_points, max(1, n_points//10)):
            ax.scatter(trajectory_pca[i, 0], trajectory_pca[i, 1],
                      c=colors[i], s=50, marker='o', alpha=0.7, zorder=3)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'{method_name} Trajectory Evolution\n(Color: Start→End)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    def _plot_comparison_trajectories(self, ax, ddpm_pca, hessian_pca):
        """Plot side-by-side trajectory comparison"""
        # DDPM trajectory
        ax.plot(ddpm_pca[:, 0], ddpm_pca[:, 1], 'r-', linewidth=3, alpha=0.8, label='DDPM Path')
        ax.scatter(ddpm_pca[0, 0], ddpm_pca[0, 1], c='green', s=150, marker='o', zorder=5)
        ax.scatter(ddpm_pca[-1, 0], ddpm_pca[-1, 1], c='red', s=150, marker='s', zorder=5)

        # Hessian-Free trajectory
        ax.plot(hessian_pca[:, 0], hessian_pca[:, 1], 'b-', linewidth=3, alpha=0.8, label='Hessian-Free Path')
        ax.scatter(hessian_pca[0, 0], hessian_pca[0, 1], c='green', s=150, marker='o', zorder=5)
        ax.scatter(hessian_pca[-1, 0], hessian_pca[-1, 1], c='red', s=150, marker='s', zorder=5)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Trajectory Comparison\n(Red: DDPM, Blue: Hessian-Free)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    def _plot_step_evolution(self, ax, ddpm_pca, hessian_pca):
        """Plot step-by-step evolution showing convergence"""
        n_steps = len(ddpm_pca)

        # Calculate distance from end point for each step
        ddpm_distances = []
        hessian_distances = []

        for i in range(n_steps):
            ddpm_dist = np.linalg.norm(ddpm_pca[i] - ddpm_pca[-1])
            hessian_dist = np.linalg.norm(hessian_pca[i] - hessian_pca[-1])
            ddpm_distances.append(ddpm_dist)
            hessian_distances.append(hessian_dist)

        # Plot convergence curves
        steps = range(n_steps)
        ax.plot(steps, ddpm_distances, 'r-', linewidth=3, alpha=0.8, label='DDPM Convergence')
        ax.plot(steps, hessian_distances, 'b-', linewidth=3, alpha=0.8, label='Hessian-Free Convergence')

        ax.set_xlabel('Sampling Step')
        ax.set_ylabel('Distance to Final Point')
        ax.set_title('Convergence Analysis\n(Distance to End Point)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    def create_trajectory_explanation_diagram(self, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """Create a diagram explaining what each line segment represents"""
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Create a simple example trajectory
        np.random.seed(42)
        n_steps = 20

        # Generate a zigzag trajectory
        x = np.linspace(0, 10, n_steps)
        y = np.sin(x) * 2 + np.random.normal(0, 0.3, n_steps)

        # Plot trajectory with color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, n_steps))

        for i in range(n_steps - 1):
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]],
                   color=colors[i], linewidth=4, alpha=0.8)
            # Add step number
            ax.text((x[i] + x[i+1])/2, (y[i] + y[i+1])/2, str(i+1),
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # Mark start and end
        ax.scatter(x[0], y[0], c='green', s=300, marker='o', label='Start (t=0)',
                  zorder=5, edgecolors='black', linewidth=2)
        ax.scatter(x[-1], y[-1], c='red', s=300, marker='s', label='End (t=T)',
                  zorder=5, edgecolors='black', linewidth=2)

        # Add arrows to show direction
        for i in range(0, n_steps-1, 3):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            ax.arrow(x[i], y[i], dx*0.3, dy*0.3, head_width=0.2, head_length=0.2,
                    fc=colors[i], ec=colors[i], alpha=0.7)

        ax.set_xlabel('PC1 (Principal Component 1)', fontsize=12)
        ax.set_ylabel('PC2 (Principal Component 2)', fontsize=12)
        ax.set_title('Trajectory Explanation: What Each Line Segment Represents',
                    fontsize=14, fontweight='bold')

        # Add explanation text
        explanation_text = """
Each line segment represents:
• One sampling step in the diffusion process
• Transition from state x_t to x_{t-1}
• Color gradient: Start (green) → End (red)
• Numbers: Sequential step numbers
• Arrows: Direction of movement
        """

        ax.text(0.02, 0.98, explanation_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Save the explanation diagram
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'trajectory_explanation_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Trajectory explanation diagram saved to: {save_path}")

        plt.show()

def main():
    """Main function to run trajectory evolution visualization"""

    print("🚀 Trajectory Evolution Visualization")
    print("="*60)
    print("Showing how sampling paths evolve from start to end")
    print("="*60)

    # Initialize visualizer
    visualizer = TrajectoryEvolutionVisualizer(n_samples=5000, num_inference_steps=25)

    # Load CIFAR-10 reference data
    X_ref = visualizer.load_cifar10_samples()

    # Perform PCA on reference data
    print("Performing PCA on CIFAR-10 data...")
    pca_model = PCA(n_components=2, svd_solver='randomized')
    pca_model.fit(X_ref)
    print(f"✓ PCA completed. Explained variance: {pca_model.explained_variance_ratio_}")

    try:
        # Generate trajectories for both methods
        methods = ['DDPM', 'Hessian_Free']
        trajectories = {}

        for method in methods:
            print(f"\n{'='*40}")
            print(f"Generating {method} trajectory...")
            print(f"{'='*40}")

            # Load pipeline
            pipe = visualizer.load_pipeline(method)

            # Generate single trajectory
            trajectory = visualizer.generate_single_trajectory(pipe, method, seed=42)
            trajectories[method] = trajectory

            # Clean up GPU memory
            del pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Create trajectory evolution plots
        print(f"\n{'='*40}")
        print("Creating trajectory evolution visualization...")
        print(f"{'='*40}")
        visualizer.plot_trajectory_evolution(
            trajectories['DDPM'],
            trajectories['Hessian_Free'],
            pca_model
        )

        # Create explanation diagram
        print(f"\n{'='*40}")
        print("Creating trajectory explanation diagram...")
        print(f"{'='*40}")
        visualizer.create_trajectory_explanation_diagram()

        print(f"\n✅ Trajectory evolution visualization completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
