#!/usr/bin/env python3
"""
DDIM vs Hessian-Free Comparison with Extreme Eigenvalues PCA
Using largest and smallest eigenvalues for PCA2 dimensionality reduction
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
from scheduler.scheduling_dpmsolver_hessian_free import DPMSolverMultistepLMSchedulerAdvanced
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

class ExtremeEigenvaluesAnalyzer:
    """Analyzer for DDIM vs Hessian-Free using extreme eigenvalues PCA"""

    def __init__(self, n_samples=5000, num_inference_steps=25, num_trajectories=100):
        self.n_samples = n_samples
        self.num_inference_steps = num_inference_steps
        self.num_trajectories = num_trajectories

        print(f"🔧 Extreme Eigenvalues Configuration:")
        print(f"   - CIFAR-10 samples for PCA: {self.n_samples}")
        print(f"   - Inference steps per trajectory: {self.num_inference_steps}")
        print(f"   - Number of trajectories per method: {self.num_trajectories}")
        print(f"   - Total sampling steps: {self.num_trajectories * self.num_inference_steps}")
        print(f"   - PCA method: Largest and Smallest eigenvalues")

    def load_pipeline(self, method_name):
        """Load pipeline for different sampling methods"""
        model_id = os.path.join(project.model_dir, 'ddpm_ema_cifar10')

        print(f"\n🔧 Loading {method_name} pipeline...")
        pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        pipe.unet.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup scheduler based on method
        if method_name == 'DDIM':
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.set_timesteps(self.num_inference_steps)
        elif method_name == 'Hessian_Free':
            pipe.scheduler = DPMSolverMultistepLMSchedulerAdvanced.from_config(pipe.scheduler.config)
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

    def create_extreme_eigenvalues_pca(self, trajectories):
        """Create PCA model using largest and smallest eigenvalues"""
        print(f"\n📊 Creating PCA model with extreme eigenvalues...")

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
        print(f"  Largest eigenvalue: {eigenvalues[0]:.6f}")
        print(f"  Smallest eigenvalue: {eigenvalues[-1]:.6f}")
        print(f"  Eigenvalue ratio (largest/smallest): {eigenvalues[0]/eigenvalues[-1]:.2f}")

        # Create custom PCA with largest and smallest eigenvalues
        print(f"  Creating custom PCA with extreme eigenvalues...")

        # Get the principal components corresponding to largest and smallest eigenvalues
        largest_pc = pca_full.components_[0]  # First principal component (largest eigenvalue)
        smallest_pc = pca_full.components_[-1]  # Last principal component (smallest eigenvalue)

        # Create custom PCA model
        class ExtremeEigenvaluesPCA:
            def __init__(self, largest_pc, smallest_pc, largest_eigenvalue, smallest_eigenvalue):
                self.components_ = np.vstack([largest_pc, smallest_pc])
                self.explained_variance_ = np.array([largest_eigenvalue, smallest_eigenvalue])
                self.explained_variance_ratio_ = self.explained_variance_ / np.sum(pca_full.explained_variance_)
                self.mean_ = pca_full.mean_

            def transform(self, X):
                # Center the data
                X_centered = X - self.mean_
                # Project onto extreme eigenvectors
                return X_centered @ self.components_.T

        pca_model = ExtremeEigenvaluesPCA(
            largest_pc, smallest_pc,
            eigenvalues[0], eigenvalues[-1]
        )

        print(f"✓ Extreme eigenvalues PCA completed.")
        print(f"  Largest eigenvalue ratio: {pca_model.explained_variance_ratio_[0]:.4f}")
        print(f"  Smallest eigenvalue ratio: {pca_model.explained_variance_ratio_[1]:.4f}")

        return pca_model, eigenvalues

    def analyze_zigzag_patterns(self, trajectories, method_name, pca_model):
        """Analyze zigzag patterns for a given method"""
        print(f"\n📈 Analyzing zigzag patterns for {method_name}...")

        # Collect all score vectors from all trajectories
        all_score_vectors = []
        all_xt_vectors = []

        for traj in trajectories:
            all_score_vectors.append(traj['score'])
            all_xt_vectors.append(traj['xt'])

        # Stack all trajectories
        score_vectors = np.vstack(all_score_vectors)
        xt_vectors = np.vstack(all_xt_vectors)

        print(f"  Total score vectors: {score_vectors.shape}")
        print(f"  Total state vectors: {xt_vectors.shape}")

        # Project to PCA space using the extreme eigenvalues PCA model
        score_pca = pca_model.transform(score_vectors)
        xt_pca = pca_model.transform(xt_vectors)

        # Compute drift vectors
        drift_vectors = np.diff(xt_vectors, axis=0)
        drift_pca = pca_model.transform(drift_vectors)

        # Analyze zigzag patterns
        angles = []
        step_lengths = []

        for i in range(len(drift_pca) - 1):
            v1 = drift_pca[i]
            v2 = drift_pca[i + 1]
            if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)

            step_lengths.append(np.linalg.norm(drift_pca[i]))

        # Compute zigzag statistics
        if angles:
            avg_angle = np.mean(angles)
            std_angle = np.std(angles)
            min_angle = np.min(angles)
            max_angle = np.max(angles)
            orthogonal_steps = np.sum(np.abs(np.array(angles) - 90) < 20)

            # Zigzag pattern score
            zigzag_score = 0
            if len(angles) > 2:
                for i in range(len(angles) - 1):
                    if (angles[i] > 90 and angles[i+1] < 90) or (angles[i] < 90 and angles[i+1] > 90):
                        zigzag_score += 1
                zigzag_score = zigzag_score / (len(angles) - 1)
        else:
            avg_angle = std_angle = min_angle = max_angle = orthogonal_steps = zigzag_score = 0

        # Step length statistics
        avg_step_length = np.mean(step_lengths)
        std_step_length = np.std(step_lengths)

        results = {
            'method_name': method_name,
            'score_pca': score_pca,
            'xt_pca': xt_pca,
            'drift_pca': drift_pca,
            'angles': angles,
            'step_lengths': step_lengths,
            'avg_angle': avg_angle,
            'std_angle': std_angle,
            'min_angle': min_angle,
            'max_angle': max_angle,
            'orthogonal_steps': orthogonal_steps,
            'zigzag_score': zigzag_score,
            'avg_step_length': avg_step_length,
            'std_step_length': std_step_length,
            'num_trajectories': len(trajectories),
            'total_steps': len(score_vectors),
            'explained_variance_ratio': pca_model.explained_variance_ratio_
        }

        print(f"  ✓ Analysis completed for {method_name}")
        print(f"    Average angle: {avg_angle:.1f}°")
        print(f"    Zigzag score: {zigzag_score:.3f}")
        print(f"    Orthogonal steps: {orthogonal_steps}/{len(angles)}")
        print(f"    Total steps analyzed: {len(score_vectors)}")

        return results

    def plot_extreme_eigenvalues_comparison(self, results_dict, eigenvalues, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """Plot DDIM vs Hessian-Free comparison using extreme eigenvalues"""
        os.makedirs(save_dir, exist_ok=True)

        methods = ['DDIM', 'Hessian_Free']
        n_methods = len(methods)

        # Create large figure with subplots
        fig, axes = plt.subplots(3, n_methods, figsize=(12, 15))
        fig.suptitle('DDIM vs Hessian-Free: Extreme Eigenvalues PCA Analysis\n(Largest vs Smallest Eigenvalues)',
                     fontsize=16, fontweight='bold')

        for i, method in enumerate(methods):
            result = results_dict[method]

            # Plot 1: State trajectory in extreme eigenvalues PCA space
            ax1 = axes[0, i]
            xt_pca = result['xt_pca']
            # Sample every 10th point for visualization
            sample_indices = np.arange(0, len(xt_pca), max(1, len(xt_pca)//1000))
            ax1.plot(xt_pca[sample_indices, 0], xt_pca[sample_indices, 1],
                    'b-', linewidth=2, alpha=0.8, label='State Trajectory')
            ax1.scatter(xt_pca[0, 0], xt_pca[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
            ax1.scatter(xt_pca[-1, 0], xt_pca[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
            ax1.set_xlabel(f'Largest Eigenvalue PC ({result["explained_variance_ratio"][0]:.1%})')
            ax1.set_ylabel(f'Smallest Eigenvalue PC ({result["explained_variance_ratio"][1]:.1%})')
            ax1.set_title(f'{method}\nState Trajectory (Extreme Eigenvalues)\nAvg Angle: {result["avg_angle"]:.1f}°')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')

            # Plot 2: Drift vectors (zigzag pattern)
            ax2 = axes[1, i]
            drift_pca = result['drift_pca']
            # Sample drift vectors for visualization
            sample_indices = np.arange(0, len(drift_pca), max(1, len(drift_pca)//500))
            for j in sample_indices:
                start_point = result['xt_pca'][j]
                end_point = start_point + drift_pca[j] * 0.15
                color = 'purple' if j % 2 == 0 else 'orange'
                ax2.arrow(start_point[0], start_point[1],
                         drift_pca[j, 0] * 0.15, drift_pca[j, 1] * 0.15,
                         head_width=0.08, head_length=0.08, fc=color, ec=color, alpha=0.7)

            ax2.plot(xt_pca[sample_indices, 0], xt_pca[sample_indices, 1],
                    'b-', linewidth=2, alpha=0.8, label='State Trajectory')
            ax2.scatter(xt_pca[0, 0], xt_pca[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
            ax2.scatter(xt_pca[-1, 0], xt_pca[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
            ax2.set_xlabel(f'Largest Eigenvalue PC ({result["explained_variance_ratio"][0]:.1%})')
            ax2.set_ylabel(f'Smallest Eigenvalue PC ({result["explained_variance_ratio"][1]:.1%})')
            ax2.set_title(f'{method}\nDrift Vectors (Extreme Eigenvalues)\nZigzag Score: {result["zigzag_score"]:.3f}')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')

            # Plot 3: Angle analysis
            ax3 = axes[2, i]
            angles = result['angles']
            if angles:
                # Sample angles for visualization
                sample_angles = angles[::max(1, len(angles)//1000)]
                ax3.plot(range(len(sample_angles)), sample_angles, 'o-', color='red',
                        linewidth=2, markersize=4, alpha=0.8)
                ax3.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='90° (Orthogonal)')
                ax3.axhline(y=result['avg_angle'], color='blue', linestyle='-', alpha=0.7,
                           label=f'Avg: {result["avg_angle"]:.1f}°')
                ax3.set_xlabel('Step')
                ax3.set_ylabel('Angle (degrees)')
                ax3.set_title(f'{method}\nAngle Analysis (Extreme Eigenvalues)\nOrthogonal Steps: {result["orthogonal_steps"]}/{len(result["angles"])}')
                ax3.legend(fontsize=10)
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No angle data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title(f'{method}\nAngle Analysis (Extreme Eigenvalues)')

        plt.tight_layout()

        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'ddim_vs_hessian_free_extreme_eigenvalues_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ DDIM vs Hessian-Free extreme eigenvalues comparison plot saved to: {save_path}")

        plt.show()

    def generate_extreme_eigenvalues_report(self, results_dict, eigenvalues, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """Generate detailed extreme eigenvalues analysis report"""
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(save_dir, f'extreme_eigenvalues_analysis_{timestamp}.txt')

        with open(report_path, 'w') as f:
            f.write("DDIM vs Hessian-Free: Extreme Eigenvalues PCA Analysis Report\n")
            f.write("="*70 + "\n\n")

            f.write("EXPERIMENTAL CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"CIFAR-10 samples for PCA: {self.n_samples}\n")
            f.write(f"Inference steps per trajectory: {self.num_inference_steps}\n")
            f.write(f"Number of trajectories per method: {self.num_trajectories}\n")
            f.write(f"Total sampling steps: {self.num_trajectories * self.num_inference_steps}\n")
            f.write(f"Total data points analyzed: {self.num_trajectories * self.num_inference_steps * 2}\n")
            f.write(f"PCA method: Extreme eigenvalues (largest and smallest)\n\n")

            f.write("EIGENVALUES ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total eigenvalues computed: {len(eigenvalues)}\n")
            f.write(f"Largest eigenvalue: {eigenvalues[0]:.6f}\n")
            f.write(f"Smallest eigenvalue: {eigenvalues[-1]:.6f}\n")
            f.write(f"Eigenvalue ratio (largest/smallest): {eigenvalues[0]/eigenvalues[-1]:.2f}\n")
            f.write(f"Condition number: {eigenvalues[0]/eigenvalues[-1]:.2f}\n\n")

            f.write("SAMPLE STATISTICS:\n")
            f.write("-" * 20 + "\n")

            for method, result in results_dict.items():
                f.write(f"\n{method} Results:\n")
                f.write(f"  - Trajectories: {result['num_trajectories']}\n")
                f.write(f"  - Total steps: {result['total_steps']}\n")
                f.write(f"  - Average angle: {result['avg_angle']:.1f}°\n")
                f.write(f"  - Angle std: {result['std_angle']:.1f}°\n")
                f.write(f"  - Min angle: {result['min_angle']:.1f}°\n")
                f.write(f"  - Max angle: {result['max_angle']:.1f}°\n")
                f.write(f"  - Zigzag score: {result['zigzag_score']:.3f}\n")
                f.write(f"  - Orthogonal steps: {result['orthogonal_steps']}/{len(result['angles'])}\n")
                f.write(f"  - Avg step length: {result['avg_step_length']:.3f}\n")
                f.write(f"  - Step length std: {result['std_step_length']:.3f}\n")
                f.write(f"  - Largest eigenvalue ratio: {result['explained_variance_ratio'][0]:.4f}\n")
                f.write(f"  - Smallest eigenvalue ratio: {result['explained_variance_ratio'][1]:.4f}\n")

            f.write(f"\nCOMPARISON SUMMARY:\n")
            f.write("-" * 20 + "\n")
            ddim_result = results_dict['DDIM']
            hessian_result = results_dict['Hessian_Free']

            f.write(f"Zigzag Score Improvement: {((ddim_result['zigzag_score'] - hessian_result['zigzag_score']) / ddim_result['zigzag_score'] * 100):.1f}%\n")
            f.write(f"Average Angle Improvement: {((ddim_result['avg_angle'] - hessian_result['avg_angle']) / ddim_result['avg_angle'] * 100):.1f}%\n")
            f.write(f"Orthogonal Steps Reduction: {((ddim_result['orthogonal_steps'] - hessian_result['orthogonal_steps']) / ddim_result['orthogonal_steps'] * 100):.1f}%\n")

            f.write(f"\nEXTREME EIGENVALUES INSIGHTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Using largest and smallest eigenvalues reveals:\n")
            f.write(f"- The most significant direction of variation (largest eigenvalue)\n")
            f.write(f"- The least significant direction of variation (smallest eigenvalue)\n")
            f.write(f"- Extreme directions that may highlight zigzag patterns\n")
            f.write(f"- Condition number indicates numerical stability\n")

        print(f"\n✓ Extreme eigenvalues analysis report saved to: {report_path}")

        # Print summary to console
        print(f"\n" + "="*80)
        print("EXTREME EIGENVALUES ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total eigenvalues computed: {len(eigenvalues)}")
        print(f"Largest eigenvalue: {eigenvalues[0]:.6f}")
        print(f"Smallest eigenvalue: {eigenvalues[-1]:.6f}")
        print(f"Eigenvalue ratio (largest/smallest): {eigenvalues[0]/eigenvalues[-1]:.2f}")
        print(f"Condition number: {eigenvalues[0]/eigenvalues[-1]:.2f}")

        print(f"\nRESULTS:")
        for method, result in results_dict.items():
            print(f"{method}:")
            print(f"  - Average angle: {result['avg_angle']:.1f}°")
            print(f"  - Zigzag score: {result['zigzag_score']:.3f}")
            print(f"  - Total steps: {result['total_steps']}")
            print(f"  - Largest eigenvalue ratio: {result['explained_variance_ratio'][0]:.4f}")
            print(f"  - Smallest eigenvalue ratio: {result['explained_variance_ratio'][1]:.4f}")

def main():
    """Main function to run DDIM vs Hessian-Free extreme eigenvalues comparison"""

    print("🚀 DDIM vs Hessian-Free: Extreme Eigenvalues PCA Analysis")
    print("="*70)
    print("Using largest and smallest eigenvalues for PCA2 dimensionality reduction")
    print("="*70)

    # Initialize analyzer
    analyzer = ExtremeEigenvaluesAnalyzer(n_samples=5000, num_inference_steps=25, num_trajectories=100)

    try:
        # Generate trajectories for both methods
        methods = ['DDIM', 'Hessian_Free']
        all_trajectories = {}

        for method in methods:
            print(f"\n{'='*60}")
            print(f"Processing Method: {method}")
            print(f"{'='*60}")

            # Load pipeline
            pipe = analyzer.load_pipeline(method)

            # Generate trajectories
            trajectories = analyzer.generate_trajectories(pipe, method, seed=42)
            all_trajectories[method] = trajectories

            # Clean up GPU memory
            del pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Create extreme eigenvalues PCA model from all trajectory data
        print(f"\n{'='*60}")
        print("Creating extreme eigenvalues PCA model...")
        print(f"{'='*60}")
        pca_model, eigenvalues = analyzer.create_extreme_eigenvalues_pca(all_trajectories)

        # Analyze each method
        results_dict = {}
        for method in methods:
            print(f"\n{'='*40}")
            print(f"Analyzing {method}...")
            print(f"{'='*40}")
            results = analyzer.analyze_zigzag_patterns(all_trajectories[method], method, pca_model)
            results_dict[method] = results

        # Generate comparison plots
        print(f"\n{'='*60}")
        print("Generating Extreme Eigenvalues Comparison Plots")
        print(f"{'='*60}")
        analyzer.plot_extreme_eigenvalues_comparison(results_dict, eigenvalues)

        # Generate extreme eigenvalues report
        analyzer.generate_extreme_eigenvalues_report(results_dict, eigenvalues)

        print(f"\n✅ DDIM vs Hessian-Free extreme eigenvalues analysis completed successfully!")
        print(f"   Analyzed {len(methods)} methods")
        print(f"   Total trajectories: {sum(r['num_trajectories'] for r in results_dict.values())}")
        print(f"   Total steps analyzed: {sum(r['total_steps'] for r in results_dict.values())}")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
