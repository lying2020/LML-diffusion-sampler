#!/usr/bin/env python3
"""
Introduction Comparison: DDIM, DPM-Solver, UniPC, PNDM (Fixed)
Comparing zigzag phenomena and anisotropy transition for introduction section
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
    'lines.linewidth': 3,
    'lines.markersize': 8,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3
})

class IntroductionComparison:
    """Comparison of DDIM, DPM-Solver, UniPC, PNDM for introduction section"""
    
    def __init__(self, n_samples=5000, num_inference_steps=25, num_trajectories=50):
        self.n_samples = n_samples
        self.num_inference_steps = num_inference_steps
        self.num_trajectories = num_trajectories
        self.methods = ['ddim', 'dpm', 'unipc', 'pndm']
        
        print(f"🔧 Introduction Comparison Configuration:")
        print(f"   - CIFAR-10 samples for PCA: {self.n_samples}")
        print(f"   - Inference steps per trajectory: {self.num_inference_steps}")
        print(f"   - Number of trajectories per method: {self.num_trajectories}")
        print(f"   - Methods: {', '.join(self.methods)}")
        
    def load_pipeline(self, method_name):
        """Load pipeline for different sampling methods"""
        model_id = os.path.join(project.model_dir, 'ddpm_ema_cifar10')
        
        print(f"\n�� Loading {method_name.upper()} pipeline...")
        pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        pipe.unet.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup scheduler based on method
        if method_name == 'ddim':
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif method_name == 'dpm':
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 2  # Use lower order for stability
            pipe.scheduler.config.algorithm_type = "dpmsolver"
        elif method_name == 'unipc':
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        elif method_name == 'pndm':
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        
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
    
    def analyze_zigzag_patterns(self, trajectories, method_name, pca_model):
        """Analyze zigzag patterns for a given method"""
        print(f"\n📈 Analyzing zigzag patterns for {method_name.upper()}...")
        
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
        
        # Project to PCA space
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
        
        print(f"  ✓ Analysis completed for {method_name.upper()}")
        print(f"    Average angle: {avg_angle:.1f}°")
        print(f"    Zigzag score: {zigzag_score:.3f}")
        print(f"    Orthogonal steps: {orthogonal_steps}/{len(angles)}")
        print(f"    Total steps analyzed: {len(score_vectors)}")
        
        return results
    
    def plot_zigzag_comparison(self, results_dict, save_dir='./zigzag_cg_hessian'):
        """Plot 4x1 zigzag comparison for introduction"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Create figure with 4 subplots in one column
        fig, axes = plt.subplots(4, 1, figsize=(8, 16))
        fig.suptitle('Zigzag Phenomenon Comparison: DDIM, DPM-Solver, UniPC, PNDM\n(PCA1 vs PCA2 - 25 Steps)', 
                     fontsize=16, fontweight='bold')
        
        # Define colors for each method
        colors = ['#E74C3C', '#3498DB', '#F39C12', '#9B59B6']  # Red, Blue, Orange, Purple
        method_names = ['DDIM', 'DPM-Solver', 'UniPC', 'PNDM']
        
        for i, (method, result) in enumerate(results_dict.items()):
            ax = axes[i]
            
            # Plot trajectory
            xt_pca = result['xt_pca']
            # Sample every 2nd point for visualization
            sample_indices = np.arange(0, len(xt_pca), max(1, len(xt_pca)//25))
            
            # Plot trajectory with color gradient
            for j in range(len(sample_indices) - 1):
                start_idx = sample_indices[j]
                end_idx = sample_indices[j + 1]
                ax.plot(xt_pca[start_idx:end_idx+1, 0], xt_pca[start_idx:end_idx+1, 1], 
                       color=colors[i], linewidth=3, alpha=0.8)
            
            # Mark start and end points
            ax.scatter(xt_pca[0, 0], xt_pca[0, 1], c='green', s=150, marker='o', 
                      label='Start', zorder=5, edgecolors='black', linewidth=2)
            ax.scatter(xt_pca[-1, 0], xt_pca[-1, 1], c='red', s=150, marker='s', 
                      label='End', zorder=5, edgecolors='black', linewidth=2)
            
            # Add step markers
            for j in range(0, len(sample_indices), max(1, len(sample_indices)//5)):
                idx = sample_indices[j]
                ax.scatter(xt_pca[idx, 0], xt_pca[idx, 1], c=colors[i], s=60, 
                          marker='o', alpha=0.7, zorder=3)
            
            ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
            ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
            ax.set_title(f'{method_names[i]}\nAvg Angle: {result["avg_angle"]:.1f}°, Zigzag Score: {result["zigzag_score"]:.3f}', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3, linewidth=1.0)
            ax.axis('equal')
            
            # Make axes labels bold
            ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'introduction_zigzag_comparison_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Introduction zigzag comparison plot saved to: {save_path}")
        
        plt.show()
    
    def plot_anisotropy_transition(self, results_dict, save_dir='./zigzag_cg_hessian'):
        """Plot 4x1 anisotropy transition comparison for introduction"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Create figure with 4 subplots in one column
        fig, axes = plt.subplots(4, 1, figsize=(8, 16))
        fig.suptitle('Anisotropy Transition: DDIM, DPM-Solver, UniPC, PNDM\n(Isotropic to Anisotropic - 25 Steps)', 
                     fontsize=16, fontweight='bold')
        
        # Define colors for each method
        colors = ['#E74C3C', '#3498DB', '#F39C12', '#9B59B6']  # Red, Blue, Orange, Purple
        method_names = ['DDIM', 'DPM-Solver', 'UniPC', 'PNDM']
        
        for i, (method, result) in enumerate(results_dict.items()):
            ax = axes[i]
            
            # Calculate anisotropy ratio for each step
            xt_pca = result['xt_pca']
            anisotropy_ratios = []
            
            for j in range(len(xt_pca)):
                if j > 0:
                    # Calculate local anisotropy ratio (PC1/PC2)
                    local_anisotropy = np.abs(xt_pca[j, 0]) / (np.abs(xt_pca[j, 1]) + 1e-8)
                    anisotropy_ratios.append(local_anisotropy)
            
            # Plot anisotropy evolution
            if anisotropy_ratios:
                steps = range(len(anisotropy_ratios))
                ax.plot(steps, anisotropy_ratios, color=colors[i], linewidth=3, alpha=0.9, 
                       label=f'{method_names[i]} Anisotropy')
                
                # Add horizontal line for isotropic reference
                ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, linewidth=2, 
                          label='Isotropic (1.0)')
                
                # Calculate and display average anisotropy
                avg_anisotropy = np.mean(anisotropy_ratios)
                ax.axhline(y=avg_anisotropy, color=colors[i], linestyle='-', alpha=0.7, 
                          label=f'Avg: {avg_anisotropy:.1f}')
            
            ax.set_xlabel('Sampling Step', fontsize=12, fontweight='bold')
            ax.set_ylabel('Anisotropy Ratio (PC1/PC2)', fontsize=12, fontweight='bold')
            ax.set_title(f'{method_names[i]}\nAnisotropy Evolution (Isotropic → Anisotropic)', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3, linewidth=1.0)
            ax.set_yscale('log')
            
            # Make axes labels bold
            ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'introduction_anisotropy_transition_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Introduction anisotropy transition plot saved to: {save_path}")
        
        plt.show()
    
    def generate_introduction_report(self, results_dict, save_dir='./zigzag_cg_hessian'):
        """Generate introduction comparison report"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(save_dir, f'introduction_comparison_{timestamp}.txt')
        
        with open(report_path, 'w') as f:
            f.write("Introduction Comparison: DDIM, DPM-Solver, UniPC, PNDM\n")
            f.write("="*70 + "\n\n")
            
            f.write("EXPERIMENTAL CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"CIFAR-10 samples for PCA: {self.n_samples}\n")
            f.write(f"Inference steps per trajectory: {self.num_inference_steps}\n")
            f.write(f"Number of trajectories per method: {self.num_trajectories}\n")
            f.write(f"Total sampling steps: {self.num_trajectories * self.num_inference_steps}\n")
            f.write(f"Methods: {', '.join(self.methods)}\n\n")
            
            f.write("ZIGZAG ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            
            for method, result in results_dict.items():
                f.write(f"\n{method.upper()} Results:\n")
                f.write(f"  - Trajectories: {result['num_trajectories']}\n")
                f.write(f"  - Total steps: {result['total_steps']}\n")
                f.write(f"  - Average angle: {result['avg_angle']:.1f}°\n")
                f.write(f"  - Angle std: {result['std_angle']:.1f}°\n")
                f.write(f"  - Zigzag score: {result['zigzag_score']:.3f}\n")
                f.write(f"  - Orthogonal steps: {result['orthogonal_steps']}/{len(result['angles'])}\n")
                f.write(f"  - Avg step length: {result['avg_step_length']:.3f}\n")
                f.write(f"  - Step length std: {result['std_step_length']:.3f}\n")
            
            f.write(f"\nCOMPARISON SUMMARY:\n")
            f.write("-" * 20 + "\n")
            
            # Find best and worst methods
            zigzag_scores = {method: result['zigzag_score'] for method, result in results_dict.items()}
            avg_angles = {method: result['avg_angle'] for method, result in results_dict.items()}
            
            best_zigzag = min(zigzag_scores, key=zigzag_scores.get)
            worst_zigzag = max(zigzag_scores, key=zigzag_scores.get)
            best_angle = min(avg_angles, key=avg_angles.get)
            worst_angle = max(avg_angles, key=avg_angles.get)
            
            f.write(f"Best zigzag score: {best_zigzag.upper()} ({zigzag_scores[best_zigzag]:.3f})\n")
            f.write(f"Worst zigzag score: {worst_zigzag.upper()} ({zigzag_scores[worst_zigzag]:.3f})\n")
            f.write(f"Best average angle: {best_angle.upper()} ({avg_angles[best_angle]:.1f}°)\n")
            f.write(f"Worst average angle: {worst_angle.upper()} ({avg_angles[worst_angle]:.1f}°)\n")
        
        print(f"\n✓ Introduction comparison report saved to: {report_path}")
        
        # Print summary to console
        print(f"\n" + "="*80)
        print("INTRODUCTION COMPARISON SUMMARY")
        print("="*80)
        
        print(f"\nZIGZAG ANALYSIS:")
        for method, result in results_dict.items():
            print(f"{method.upper()}:")
            print(f"  - Average angle: {result['avg_angle']:.1f}°")
            print(f"  - Zigzag score: {result['zigzag_score']:.3f}")
            print(f"  - Total steps: {result['total_steps']}")

def main():
    """Main function to run introduction comparison"""
    
    print("🚀 Introduction Comparison: DDIM, DPM-Solver, UniPC, PNDM (Fixed)")
    print("="*70)
    print("Comparing zigzag phenomena and anisotropy transition for introduction section")
    print("="*70)
    
    # Initialize analyzer
    analyzer = IntroductionComparison(n_samples=5000, num_inference_steps=25, num_trajectories=50)
    
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
        
        # Create PCA model from all trajectory data
        print(f"\n{'='*60}")
        print("Creating PCA model from all trajectory data...")
        print(f"{'='*60}")
        pca_model = analyzer.create_pca_from_trajectories(all_trajectories)
        
        # Analyze each method
        results_dict = {}
        for method in analyzer.methods:
            print(f"\n{'='*40}")
            print(f"Analyzing {method.upper()}...")
            print(f"{'='*40}")
            results = analyzer.analyze_zigzag_patterns(all_trajectories[method], method, pca_model)
            results_dict[method] = results
        
        # Generate comparison plots
        print(f"\n{'='*60}")
        print("Generating Introduction Comparison Plots")
        print(f"{'='*60}")
        
        # Plot 1: Zigzag comparison
        analyzer.plot_zigzag_comparison(results_dict)
        
        # Plot 2: Anisotropy transition
        analyzer.plot_anisotropy_transition(results_dict)
        
        # Generate report
        analyzer.generate_introduction_report(results_dict)
        
        print(f"\n✅ Introduction comparison completed successfully!")
        print(f"   Analyzed {len(analyzer.methods)} methods")
        print(f"   Total trajectories: {sum(r['num_trajectories'] for r in results_dict.values())}")
        print(f"   Total steps analyzed: {sum(r['total_steps'] for r in results_dict.values())}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
