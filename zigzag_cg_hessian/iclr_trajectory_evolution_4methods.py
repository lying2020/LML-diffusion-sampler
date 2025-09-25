#!/usr/bin/env python3
"""
ICLR Paper Format: 4-Method Trajectory Evolution Visualization
DDIM, PNDM, UniPC, DDPM-Solver trajectory evolution comparison
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
from diffusers import DDPMPipeline, DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler
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

class ICLRTrajectoryEvolution4Methods:
    """ICLR paper format trajectory evolution for 4 methods"""

    def __init__(self, n_samples=5000, num_inference_steps=25, num_trajectories=50):
        self.n_samples = n_samples
        self.num_inference_steps = num_inference_steps
        self.num_trajectories = num_trajectories
        self.methods = ['ddim', 'pndm', 'unipc', 'dpm']

        print(f"🔧 ICLR 4-Method Trajectory Evolution Configuration:")
        print(f"   - CIFAR-10 samples for PCA: {self.n_samples}")
        print(f"   - Inference steps per trajectory: {self.num_inference_steps}")
        print(f"   - Number of trajectories per method: {self.num_trajectories}")
        print(f"   - Methods: {', '.join(self.methods)}")

    def load_pipeline(self, method_name):
        """Load pipeline for different sampling methods"""
        model_id = os.path.join(project.model_dir, 'ddpm_ema_cifar10')

        print(f"\n🔧 Loading {method_name.upper()} pipeline...")
        pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        pipe.unet.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup scheduler based on method
        if method_name == 'ddim':
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif method_name == 'pndm':
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        elif method_name == 'unipc':
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        elif method_name == 'dpm':
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 2  # Use lower order for stability
            pipe.scheduler.config.algorithm_type = "dpmsolver"

        pipe.scheduler.set_timesteps(self.num_inference_steps)
        print(f"✓ {method_name.upper()} pipeline loaded successfully")
        return pipe

    def generate_trajectories(self, pipe, method_name, seed=42):
        """Generate multiple trajectories for statistical analysis"""
        print(f"\n🚀 Generating {self.num_trajectories} trajectories using {method_name.upper()}...")

        trajectories = []
        torch.manual_seed(seed)

        for i in range(self.num_trajectories):
            if i % 10 == 0:
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

            # Sampling step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Convert to numpy arrays
        for key in ['xt', 'score', 'timesteps', 'noise_pred']:
            trajectory_data[key] = np.array(trajectory_data[key])

        return trajectory_data

    def create_pca_from_trajectories(self, trajectories):
        """Create PCA model from trajectory data"""
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

    def plot_iclr_trajectory_evolution_4methods(self, trajectories, pca_model, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """Plot ICLR paper format trajectory evolution for 4 methods - 1x4 layout"""
        os.makedirs(save_dir, exist_ok=True)

        # Create figure with 4 subplots in one row - optimized for ICLR paper format
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # Single row, 4 columns
        fig.suptitle('Trajectory Evolution: DDIM, PNDM, UniPC, DDPM-Solver\n(Noise → Image - 25 Steps)',
                     fontsize=18, fontweight='bold', y=0.95)

        # Define colors for ICLR paper format
        method_colors = {
            'ddim': '#E74C3C',      # Red
            'pndm': '#9B59B6',      # Purple
            'unipc': '#F39C12',     # Orange
            'dpm': '#3498DB'        # Blue
        }

        method_names = {
            'ddim': 'DDIM',
            'pndm': 'PNDM',
            'unipc': 'UniPC',
            'dpm': 'DDPM-Solver'
        }

        start_color = '#27AE60'  # Green
        end_color = '#E67E22'    # Orange

        for i, (method, traj_list) in enumerate(trajectories.items()):
            ax = axes[i]

            # Use first trajectory for visualization
            traj = traj_list[0]
            xt_pca = pca_model.transform(traj['xt'])

            # Create color gradient from start to end
            n_points = len(xt_pca)
            colors = plt.cm.viridis(np.linspace(0, 1, n_points))

            # Plot trajectory with thick lines and color gradient
            for j in range(n_points - 1):
                ax.plot([xt_pca[j, 0], xt_pca[j+1, 0]],
                       [xt_pca[j, 1], xt_pca[j+1, 1]],
                       color=colors[j], linewidth=4, alpha=0.9)

            # Mark start and end points with larger markers
            ax.scatter(xt_pca[0, 0], xt_pca[0, 1],
                      c=start_color, s=200, marker='o', label='Start', zorder=5,
                      edgecolors='black', linewidth=2)
            ax.scatter(xt_pca[-1, 0], xt_pca[-1, 1],
                      c=end_color, s=200, marker='s', label='End', zorder=5,
                      edgecolors='black', linewidth=2)

            # Add step markers with larger size
            for j in range(0, n_points, max(1, n_points//8)):
                ax.scatter(xt_pca[j, 0], xt_pca[j, 1],
                          c=colors[j], s=80, marker='o', alpha=0.8, zorder=3)

            # Add method-specific color line for legend
            ax.plot([], [], color=method_colors[method], linewidth=4,
                   label=f'{method_names[method]} Path', alpha=0.8)

            ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
            ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
            ax.set_title(f'{method_names[method]}\nTrajectory Evolution', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3, linewidth=1.0)
            ax.axis('equal')

            # Make axes labels bold
            ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)

        # Adjust layout for ICLR paper format
        plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space for main title

        # Save the plot with high DPI for ICLR paper
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'iclr_trajectory_evolution_4methods_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\n✓ ICLR format 4-method trajectory evolution plot saved to: {save_path}")

        plt.show()

    def generate_4methods_report(self, trajectories, pca_model, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """Generate 4-method trajectory evolution report"""
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(save_dir, f'iclr_4methods_trajectory_{timestamp}.txt')

        with open(report_path, 'w') as f:
            f.write("ICLR 4-Method Trajectory Evolution Analysis\n")
            f.write("="*70 + "\n\n")

            f.write("EXPERIMENTAL CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"CIFAR-10 samples for PCA: {self.n_samples}\n")
            f.write(f"Inference steps per trajectory: {self.num_inference_steps}\n")
            f.write(f"Number of trajectories per method: {self.num_trajectories}\n")
            f.write(f"Total sampling steps: {self.num_trajectories * self.num_inference_steps}\n")
            f.write(f"Methods: {', '.join(self.methods)}\n\n")

            f.write("PCA ANALYSIS:\n")
            f.write("-" * 15 + "\n")
            f.write(f"PC1 explained variance: {pca_model.explained_variance_ratio_[0]:.4f}\n")
            f.write(f"PC2 explained variance: {pca_model.explained_variance_ratio_[1]:.4f}\n")
            f.write(f"Total explained variance: {np.sum(pca_model.explained_variance_ratio_):.4f}\n\n")

            f.write("TRAJECTORY ANALYSIS:\n")
            f.write("-" * 20 + "\n")

            for method, traj_list in trajectories.items():
                f.write(f"\n{method.upper()} Results:\n")
                f.write(f"  - Trajectories: {len(traj_list)}\n")
                f.write(f"  - Steps per trajectory: {len(traj_list[0]['xt'])}\n")
                f.write(f"  - Total data points: {len(traj_list) * len(traj_list[0]['xt'])}\n")

                # Calculate trajectory statistics
                all_xt = np.vstack([traj['xt'] for traj in traj_list])
                all_xt_pca = pca_model.transform(all_xt)

                # Calculate trajectory length
                traj_lengths = []
                for traj in traj_list:
                    traj_pca = pca_model.transform(traj['xt'])
                    length = np.sum(np.linalg.norm(np.diff(traj_pca, axis=0), axis=1))
                    traj_lengths.append(length)

                f.write(f"  - Average trajectory length: {np.mean(traj_lengths):.2f}\n")
                f.write(f"  - Trajectory length std: {np.std(traj_lengths):.2f}\n")
                f.write(f"  - Min trajectory length: {np.min(traj_lengths):.2f}\n")
                f.write(f"  - Max trajectory length: {np.max(traj_lengths):.2f}\n")

        print(f"\n✓ 4-method trajectory evolution report saved to: {report_path}")

        # Print summary to console
        print(f"\n" + "="*80)
        print("ICLR 4-METHOD TRAJECTORY EVOLUTION SUMMARY")
        print("="*80)

        print(f"\nPCA ANALYSIS:")
        print(f"PC1 explained variance: {pca_model.explained_variance_ratio_[0]:.4f}")
        print(f"PC2 explained variance: {pca_model.explained_variance_ratio_[1]:.4f}")
        print(f"Total explained variance: {np.sum(pca_model.explained_variance_ratio_):.4f}")

        print(f"\nTRAJECTORY ANALYSIS:")
        for method, traj_list in trajectories.items():
            print(f"{method.upper()}:")
            print(f"  - Trajectories: {len(traj_list)}")
            print(f"  - Steps per trajectory: {len(traj_list[0]['xt'])}")

def main():
    """Main function to run ICLR 4-method trajectory evolution"""

    print("🚀 ICLR 4-Method Trajectory Evolution Visualization")
    print("="*70)
    print("DDIM, PNDM, UniPC, DDPM-Solver trajectory evolution comparison")
    print("="*70)

    # Initialize visualizer
    visualizer = ICLRTrajectoryEvolution4Methods(n_samples=5000, num_inference_steps=25, num_trajectories=50)

    try:
        # Generate trajectories for all methods
        all_trajectories = {}

        for method in visualizer.methods:
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

        # Create ICLR paper format trajectory evolution plots
        print(f"\n{'='*60}")
        print("Creating ICLR Paper Format 4-Method Trajectory Evolution...")
        print(f"{'='*60}")
        visualizer.plot_iclr_trajectory_evolution_4methods(all_trajectories, pca_model)

        # Generate report
        visualizer.generate_4methods_report(all_trajectories, pca_model)

        print(f"\n✅ ICLR 4-method trajectory evolution completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
