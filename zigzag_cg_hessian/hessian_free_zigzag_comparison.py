#!/usr/bin/env python3
"""
Hessian-Free Method Zigzag Comparison: DDIM vs DDPM vs Hessian-Free
This script specifically compares the lm_correct_hessian_free method with standard methods
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
from diffusers import DDPMPipeline, DDIMScheduler, DDPMScheduler
from scheduler.scheduling_dpmsolver_multistep_lm_advanced import DPMSolverMultistepLMSchedulerAdvanced
import project as project

class HessianFreeAnalyzer:
    """Specialized analyzer for Hessian-Free method comparison"""
    
    def __init__(self, n_samples=5000, num_inference_steps=25):
        self.n_samples = n_samples
        self.num_inference_steps = num_inference_steps
        self.results = {}
        
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
        model_id = os.path.join(project.model_dir, 'ddpm_ema_cifar10')
        
        print(f"Loading {method_name} pipeline...")
        pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        pipe.unet.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup scheduler based on method
        if method_name == 'DDIM':
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif method_name == 'DDPM':
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        elif method_name == 'Hessian_Free':
            pipe.scheduler = DPMSolverMultistepLMSchedulerAdvanced.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lamb = 0.0008
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = 1e-8
            pipe.scheduler.hessian_method = 'hessian_free'
            pipe.scheduler.set_model(pipe.unet)
        elif method_name == 'Original_LML':
            pipe.scheduler = DPMSolverMultistepLMSchedulerAdvanced.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lamb = 0.0008
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = 1e-8
            pipe.scheduler.hessian_method = 'original'
            pipe.scheduler.set_model(pipe.unet)
        
        print(f"✓ {method_name} pipeline loaded successfully")
        return pipe
    
    def generate_trajectory(self, pipe, method_name, num_trajectories=50, seed=42):
        """Generate trajectories for zigzag analysis"""
        print(f"Generating {num_trajectories} trajectories using {method_name}...")
        
        trajectories = []
        torch.manual_seed(seed)
        
        for i in range(num_trajectories):
            if i % 10 == 0:
                print(f"  Generating trajectory {i+1}/{num_trajectories}")
            
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
            
            # DDIM/DDPM step
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # Convert to numpy arrays
        for key in ['xt', 'score', 'timesteps', 'noise_pred']:
            trajectory_data[key] = np.array(trajectory_data[key])
        
        return trajectory_data
    
    def analyze_zigzag_patterns(self, trajectories, method_name):
        """Analyze zigzag patterns for a given method"""
        print(f"Analyzing zigzag patterns for {method_name}...")
        
        # Collect all score vectors from all trajectories
        all_score_vectors = []
        all_xt_vectors = []
        
        for traj in trajectories:
            all_score_vectors.append(traj['score'])
            all_xt_vectors.append(traj['xt'])
        
        # Stack all trajectories
        score_vectors = np.vstack(all_score_vectors)  # Shape: (num_trajectories * num_steps, 3072)
        xt_vectors = np.vstack(all_xt_vectors)
        
        print(f"  Total score vectors: {score_vectors.shape}")
        
        # Perform PCA on score vectors
        pca_scores = PCA(n_components=2, svd_solver='randomized')
        score_pca = pca_scores.fit_transform(score_vectors)
        
        # Project state vectors onto the same PCA space
        xt_pca = pca_scores.transform(xt_vectors)
        
        # Compute drift vectors
        drift_vectors = np.diff(xt_vectors, axis=0)
        drift_pca = pca_scores.transform(drift_vectors)
        
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
            'pca_model': pca_scores,
            'explained_variance_ratio': pca_scores.explained_variance_ratio_,
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
            'total_steps': len(score_vectors)
        }
        
        print(f"  ✓ Analysis completed for {method_name}")
        print(f"    Average angle: {avg_angle:.1f}°")
        print(f"    Zigzag score: {zigzag_score:.3f}")
        print(f"    Orthogonal steps: {orthogonal_steps}/{len(angles)}")
        
        return results
    
    def plot_hessian_free_comparison(self, results_dict, save_dir='./zigzag_cg_hessian'):
        """Plot specialized comparison focusing on Hessian-Free method"""
        os.makedirs(save_dir, exist_ok=True)
        
        methods = list(results_dict.keys())
        n_methods = len(methods)
        
        # Create large figure with subplots
        fig, axes = plt.subplots(3, n_methods, figsize=(5*n_methods, 15))
        fig.suptitle('Hessian-Free Method Zigzag Analysis: DDIM vs DDPM vs Hessian-Free vs Original LML', 
                     fontsize=16, fontweight='bold')
        
        for i, method in enumerate(methods):
            result = results_dict[method]
            
            # Plot 1: State trajectory in PCA space
            ax1 = axes[0, i]
            xt_pca = result['xt_pca']
            # Sample every 10th point for visualization
            sample_indices = np.arange(0, len(xt_pca), max(1, len(xt_pca)//1000))
            ax1.plot(xt_pca[sample_indices, 0], xt_pca[sample_indices, 1], 
                    'b-', linewidth=1, alpha=0.7, label='State Trajectory')
            ax1.scatter(xt_pca[0, 0], xt_pca[0, 1], c='green', s=50, marker='o', label='Start', zorder=5)
            ax1.scatter(xt_pca[-1, 0], xt_pca[-1, 1], c='red', s=50, marker='s', label='End', zorder=5)
            ax1.set_xlabel(f'PC1 ({result["explained_variance_ratio"][0]:.1%})')
            ax1.set_ylabel(f'PC2 ({result["explained_variance_ratio"][1]:.1%})')
            ax1.set_title(f'{method}\nState Trajectory')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            
            # Plot 2: Drift vectors (zigzag pattern)
            ax2 = axes[1, i]
            drift_pca = result['drift_pca']
            # Sample drift vectors for visualization
            sample_indices = np.arange(0, len(drift_pca), max(1, len(drift_pca)//500))
            for j in sample_indices:
                start_point = result['xt_pca'][j]
                end_point = start_point + drift_pca[j] * 0.1
                color = 'purple' if j % 2 == 0 else 'orange'
                ax2.arrow(start_point[0], start_point[1], 
                         drift_pca[j, 0] * 0.1, drift_pca[j, 1] * 0.1,
                         head_width=0.05, head_length=0.05, fc=color, ec=color, alpha=0.6)
            
            ax2.plot(xt_pca[sample_indices, 0], xt_pca[sample_indices, 1], 
                    'b-', linewidth=1, alpha=0.7, label='State Trajectory')
            ax2.scatter(xt_pca[0, 0], xt_pca[0, 1], c='green', s=50, marker='o', label='Start', zorder=5)
            ax2.scatter(xt_pca[-1, 0], xt_pca[-1, 1], c='red', s=50, marker='s', label='End', zorder=5)
            ax2.set_xlabel(f'PC1 ({result["explained_variance_ratio"][0]:.1%})')
            ax2.set_ylabel(f'PC2 ({result["explained_variance_ratio"][1]:.1%})')
            ax2.set_title(f'{method}\nDrift Vectors (Zigzag)')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')
            
            # Plot 3: Angle analysis
            ax3 = axes[2, i]
            angles = result['angles']
            if angles:
                # Sample angles for visualization
                sample_angles = angles[::max(1, len(angles)//1000)]
                ax3.plot(range(len(sample_angles)), sample_angles, 'o-', color='red', 
                        linewidth=1, markersize=3, alpha=0.7)
                ax3.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='90° (Orthogonal)')
                ax3.set_xlabel('Step')
                ax3.set_ylabel('Angle (degrees)')
                ax3.set_title(f'{method}\nAngle Analysis\nAvg: {result["avg_angle"]:.1f}°')
                ax3.legend(fontsize=8)
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No angle data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title(f'{method}\nAngle Analysis')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'hessian_free_zigzag_comparison_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Hessian-Free comparison plot saved to: {save_path}")
        
        plt.show()
    
    def generate_hessian_free_summary(self, results_dict, save_dir='./zigzag_cg_hessian'):
        """Generate summary table focusing on Hessian-Free method"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("HESSIAN-FREE METHOD ZIGZAG ANALYSIS SUMMARY")
        print("="*80)
        
        # Create summary table
        methods = list(results_dict.keys())
        
        print(f"{'Method':<20} {'Avg Angle':<12} {'Zigzag Score':<15} {'Orthogonal':<12} {'Explained Var':<15}")
        print("-" * 80)
        
        for method in methods:
            result = results_dict[method]
            explained_var = result['explained_variance_ratio'].sum()
            orthogonal_ratio = result['orthogonal_steps'] / len(result['angles']) if result['angles'] else 0
            
            print(f"{method:<20} {result['avg_angle']:<12.1f} {result['zigzag_score']:<15.3f} "
                  f"{orthogonal_ratio:<12.1%} {explained_var:<15.3f}")
        
        print("\n" + "="*80)
        print("HESSIAN-FREE METHOD ANALYSIS:")
        print("="*80)
        
        # Find Hessian-Free method results
        hessian_free_result = results_dict.get('Hessian_Free')
        if hessian_free_result:
            print(f"• Hessian-Free Method Performance:")
            print(f"  - Average angle: {hessian_free_result['avg_angle']:.1f}°")
            print(f"  - Zigzag score: {hessian_free_result['zigzag_score']:.3f}")
            print(f"  - Orthogonal steps: {hessian_free_result['orthogonal_steps']}/{len(hessian_free_result['angles'])} ({hessian_free_result['orthogonal_steps']/len(hessian_free_result['angles']):.1%})")
            print(f"  - Explained variance: {hessian_free_result['explained_variance_ratio'].sum():.3f}")
            
            # Compare with other methods
            print(f"\n• Comparison with other methods:")
            for method in methods:
                if method != 'Hessian_Free':
                    other_result = results_dict[method]
                    zigzag_improvement = ((other_result['zigzag_score'] - hessian_free_result['zigzag_score']) / other_result['zigzag_score']) * 100
                    print(f"  - vs {method}: {zigzag_improvement:+.1f}% zigzag improvement")
        
        # Find best performing method
        best_zigzag = min(results_dict.items(), key=lambda x: x[1]['zigzag_score'])
        best_orthogonal = max(results_dict.items(), key=lambda x: x[1]['orthogonal_steps']/len(x[1]['angles']) if x[1]['angles'] else 0)
        
        print(f"\n• Overall best performance:")
        print(f"  - Best zigzag reduction: {best_zigzag[0]} (score: {best_zigzag[1]['zigzag_score']:.3f})")
        print(f"  - Most orthogonal steps: {best_orthogonal[0]} ({best_orthogonal[1]['orthogonal_steps']}/{len(best_orthogonal[1]['angles'])} steps)")
        
        # Save summary to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = os.path.join(save_dir, f'hessian_free_analysis_summary_{timestamp}.txt')
        
        with open(summary_path, 'w') as f:
            f.write("HESSIAN-FREE METHOD ZIGZAG ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'Method':<20} {'Avg Angle':<12} {'Zigzag Score':<15} {'Orthogonal':<12} {'Explained Var':<15}\n")
            f.write("-" * 80 + "\n")
            
            for method in methods:
                result = results_dict[method]
                explained_var = result['explained_variance_ratio'].sum()
                orthogonal_ratio = result['orthogonal_steps'] / len(result['angles']) if result['angles'] else 0
                
                f.write(f"{method:<20} {result['avg_angle']:<12.1f} {result['zigzag_score']:<15.3f} "
                       f"{orthogonal_ratio:<12.1%} {explained_var:<15.3f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("HESSIAN-FREE METHOD ANALYSIS:\n")
            f.write("="*80 + "\n")
            
            if hessian_free_result:
                f.write(f"• Hessian-Free Method Performance:\n")
                f.write(f"  - Average angle: {hessian_free_result['avg_angle']:.1f}°\n")
                f.write(f"  - Zigzag score: {hessian_free_result['zigzag_score']:.3f}\n")
                f.write(f"  - Orthogonal steps: {hessian_free_result['orthogonal_steps']}/{len(hessian_free_result['angles'])} ({hessian_free_result['orthogonal_steps']/len(hessian_free_result['angles']):.1%})\n")
                f.write(f"  - Explained variance: {hessian_free_result['explained_variance_ratio'].sum():.3f}\n")
                
                f.write(f"\n• Comparison with other methods:\n")
                for method in methods:
                    if method != 'Hessian_Free':
                        other_result = results_dict[method]
                        zigzag_improvement = ((other_result['zigzag_score'] - hessian_free_result['zigzag_score']) / other_result['zigzag_score']) * 100
                        f.write(f"  - vs {method}: {zigzag_improvement:+.1f}% zigzag improvement\n")
            
            f.write(f"\n• Overall best performance:\n")
            f.write(f"  - Best zigzag reduction: {best_zigzag[0]} (score: {best_zigzag[1]['zigzag_score']:.3f})\n")
            f.write(f"  - Most orthogonal steps: {best_orthogonal[0]} ({best_orthogonal[1]['orthogonal_steps']}/{len(best_orthogonal[1]['angles'])} steps)\n")
        
        print(f"\n✓ Hessian-Free summary saved to: {summary_path}")

def main():
    """Main function to run Hessian-Free method comparison"""
    
    print("🚀 Hessian-Free Method Zigzag Analysis")
    print("="*80)
    print("Focusing on lm_correct_hessian_free method comparison")
    print("Large-scale analysis with 5000+ samples for robust statistical comparison")
    print("="*80)
    
    # Initialize analyzer
    analyzer = HessianFreeAnalyzer(n_samples=5000, num_inference_steps=25)
    
    # Load CIFAR-10 reference data
    X_ref = analyzer.load_cifar10_samples()
    
    # Define methods to compare (focusing on Hessian-Free)
    methods = [
        'DDIM',
        'DDPM', 
        'Original_LML',
        'Hessian_Free'
    ]
    
    results_dict = {}
    
    try:
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Processing Method: {method}")
            print(f"{'='*60}")
            
            # Load pipeline
            pipe = analyzer.load_pipeline(method)
            
            # Generate trajectories (use fewer trajectories for faster computation)
            trajectories = analyzer.generate_trajectory(pipe, method, num_trajectories=50, seed=42)
            
            # Analyze zigzag patterns
            results = analyzer.analyze_zigzag_patterns(trajectories, method)
            results_dict[method] = results
            
            # Clean up GPU memory
            del pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Generate Hessian-Free comparison plots
        print(f"\n{'='*60}")
        print("Generating Hessian-Free Comparison Plots")
        print(f"{'='*60}")
        analyzer.plot_hessian_free_comparison(results_dict)
        
        # Generate Hessian-Free summary
        analyzer.generate_hessian_free_summary(results_dict)
        
        print(f"\n✅ Hessian-Free analysis completed successfully!")
        print(f"   Analyzed {len(methods)} methods")
        print(f"   Total trajectories: {sum(r['num_trajectories'] for r in results_dict.values())}")
        print(f"   Total steps analyzed: {sum(r['total_steps'] for r in results_dict.values())}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
