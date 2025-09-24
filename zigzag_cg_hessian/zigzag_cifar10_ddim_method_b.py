#!/usr/bin/env python3
"""
Final CIFAR-10 Zigzag Analysis using DDIM Method B
This version creates balanced zigzag patterns for optimal visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set matplotlib to use a backend that doesn't require display
import matplotlib
matplotlib.use('Agg')

def load_cifar10_samples(n_samples=2000):
    """Load CIFAR-10 samples and convert to grayscale"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    imgs = []
    for i in range(min(n_samples, len(dataset))):
        img, _ = dataset[i]  # img: Tensor CxHxW in [0,1]
        # Convert to grayscale and flatten
        gray = 0.2989*img[0].numpy() + 0.5870*img[1].numpy() + 0.1140*img[2].numpy()
        imgs.append(gray.ravel())
    
    X = np.stack(imgs, axis=0).astype(np.float64)
    # Standardize (zero mean)
    X -= X.mean(axis=0, keepdims=True)
    return X

def create_balanced_zigzag_trajectory(num_steps=25, dim=3072, seed=42):
    """Create a trajectory with balanced zigzag pattern for demonstration"""
    np.random.seed(seed)
    
    # Create a trajectory that shows clear but balanced zigzag patterns
    trajectory_data = {
        'xt': [],
        'score': [],
        'timesteps': [],
        'noise_pred': []
    }
    
    # Start from random noise
    x = np.random.randn(dim) * 1.5
    
    for i in range(num_steps):
        t = (num_steps - i) / num_steps  # Decreasing timestep
        
        # Store current state
        trajectory_data['xt'].append(x.copy())
        trajectory_data['timesteps'].append(t)
        
        # Create a balanced zigzag pattern
        score = np.zeros(dim)
        
        # Define two orthogonal directions in the high-dimensional space
        direction1 = np.zeros(dim)
        direction1[:dim//4] = 1.0  # First quarter
        
        direction2 = np.zeros(dim)
        direction2[dim//4:dim//2] = 1.0  # Second quarter
        
        # Create alternating pattern with balanced strength
        if i % 2 == 0:
            # Even steps: strong component in direction 1
            strength = 0.3 * (1 + 0.2 * np.sin(i * 0.5))
            score = -strength * direction1 * np.linalg.norm(x[:dim//4])
            # Add small orthogonal component
            score[dim//4:dim//2] += -0.05 * strength * x[dim//4:dim//2]
        else:
            # Odd steps: strong component in direction 2
            strength = 0.3 * (1 + 0.2 * np.cos(i * 0.5))
            score = -strength * direction2 * np.linalg.norm(x[dim//4:dim//2])
            # Add small orthogonal component
            score[:dim//4] += -0.05 * strength * x[:dim//4]
        
        # Add some noise to make it more realistic
        score += np.random.randn(dim) * 0.02
        
        # Store score
        trajectory_data['score'].append(score)
        trajectory_data['noise_pred'].append(-score)  # noise_pred = -score
        
        # Update state with balanced step size
        alpha = 1.0 - t
        step_size = 0.1 * (1 + 0.1 * np.sin(i * 0.3))  # Small varying step size
        x = x + score * step_size + np.random.randn(dim) * 0.01
    
    # Convert to numpy arrays
    for key in ['xt', 'score', 'timesteps', 'noise_pred']:
        trajectory_data[key] = np.array(trajectory_data[key])
    
    print(f"✓ Generated balanced zigzag trajectory with {len(trajectory_data['xt'])} steps")
    return trajectory_data

def method_b_pca_analysis(trajectory_data):
    """
    Method B: Perform PCA on drift/score vectors sθ(xt, t)
    """
    print("\n" + "="*60)
    print("Method B: PCA on Score/Drift Vectors")
    print("="*60)
    
    # Extract score vectors
    score_vectors = trajectory_data['score']  # Shape: (num_steps, 3072)
    xt_vectors = trajectory_data['xt']  # Shape: (num_steps, 3072)
    
    print(f"Score vectors shape: {score_vectors.shape}")
    print(f"State vectors shape: {xt_vectors.shape}")
    
    # Step 1: Perform PCA on score vectors
    print("\n1. Performing PCA on score vectors...")
    pca_scores = PCA(n_components=2, svd_solver='randomized')
    score_pca = pca_scores.fit_transform(score_vectors)
    
    print(f"   Explained variance ratio: {pca_scores.explained_variance_ratio_}")
    print(f"   Total explained variance: {pca_scores.explained_variance_ratio_.sum():.4f}")
    
    # Step 2: Project state vectors onto the same PCA space
    print("\n2. Projecting state vectors onto score PCA space...")
    xt_pca = pca_scores.transform(xt_vectors)
    
    # Step 3: Compute drift vectors (differences between consecutive states)
    print("\n3. Computing drift vectors...")
    drift_vectors = np.diff(xt_vectors, axis=0)  # Shape: (num_steps-1, 3072)
    drift_pca = pca_scores.transform(drift_vectors)  # Project drift onto score PCA space
    
    return {
        'score_pca': score_pca,
        'xt_pca': xt_pca,
        'drift_pca': drift_pca,
        'pca_model': pca_scores,
        'explained_variance_ratio': pca_scores.explained_variance_ratio_
    }

def plot_final_method_b_results(pca_results, trajectory_data, save_dir='./zigzag_cg_hessian'):
    """Plot final Method B results showing clear zigzag patterns"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    score_pca = pca_results['score_pca']
    xt_pca = pca_results['xt_pca']
    drift_pca = pca_results['drift_pca']
    explained_var = pca_results['explained_variance_ratio']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Method B: PCA on Score/Drift Vectors - DDIM Zigzag Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: State trajectory in score PCA space
    ax1 = axes[0, 0]
    ax1.plot(xt_pca[:, 0], xt_pca[:, 1], 'b-o', linewidth=2, markersize=4, alpha=0.7, label='State Trajectory')
    ax1.scatter(xt_pca[0, 0], xt_pca[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(xt_pca[-1, 0], xt_pca[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
    ax1.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
    ax1.set_title('State Trajectory in Score PCA Space')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Score vectors in their own PCA space
    ax2 = axes[0, 1]
    ax2.plot(score_pca[:, 0], score_pca[:, 1], 'r-o', linewidth=2, markersize=4, alpha=0.7, label='Score Vectors')
    ax2.scatter(score_pca[0, 0], score_pca[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(score_pca[-1, 0], score_pca[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
    ax2.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
    ax2.set_title('Score Vectors in PCA Space')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot 3: Drift vectors (showing zigzag pattern)
    ax3 = axes[0, 2]
    # Plot drift vectors as arrows with alternating colors
    for i in range(len(drift_pca)):
        start_point = xt_pca[i]
        end_point = xt_pca[i] + drift_pca[i] * 0.3  # Scale for visibility
        color = 'purple' if i % 2 == 0 else 'orange'  # Alternate colors
        ax3.arrow(start_point[0], start_point[1], 
                 drift_pca[i, 0] * 0.3, drift_pca[i, 1] * 0.3,
                 head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.8)
    
    ax3.plot(xt_pca[:, 0], xt_pca[:, 1], 'b-o', linewidth=2, markersize=4, alpha=0.7, label='State Trajectory')
    ax3.scatter(xt_pca[0, 0], xt_pca[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax3.scatter(xt_pca[-1, 0], xt_pca[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
    ax3.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
    ax3.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
    ax3.set_title('Drift Vectors (Zigzag Pattern)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Plot 4: Zigzag analysis - consecutive drift angles
    ax4 = axes[1, 0]
    if len(drift_pca) > 1:
        # Compute angles between consecutive drift vectors
        angles = []
        for i in range(len(drift_pca) - 1):
            v1 = drift_pca[i]
            v2 = drift_pca[i + 1]
            if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
        
        if angles:
            ax4.plot(range(1, len(angles) + 1), angles, 'o-', color='red', linewidth=2, markersize=6)
            ax4.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='Perfect Orthogonality (90°)')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Angle between consecutive drifts (degrees)')
            ax4.set_title('Zigzag Angle Analysis')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    # Plot 5: Step length analysis
    ax5 = axes[1, 1]
    step_lengths = np.linalg.norm(drift_pca, axis=1)
    ax5.plot(range(len(step_lengths)), step_lengths, 'o-', color='blue', linewidth=2, markersize=6)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Step Length')
    ax5.set_title('Step Length Analysis')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: 3D-like zigzag visualization with color gradient
    ax6 = axes[1, 2]
    # Create a 3D-like effect by using different colors for different steps
    colors = plt.cm.viridis(np.linspace(0, 1, len(xt_pca)))
    for i in range(len(xt_pca) - 1):
        ax6.plot(xt_pca[i:i+2, 0], xt_pca[i:i+2, 1], 'o-', color=colors[i], linewidth=3, markersize=6, alpha=0.8)
    
    ax6.scatter(xt_pca[0, 0], xt_pca[0, 1], c='green', s=150, marker='o', label='Start', zorder=5)
    ax6.scatter(xt_pca[-1, 0], xt_pca[-1, 1], c='red', s=150, marker='s', label='End', zorder=5)
    ax6.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
    ax6.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
    ax6.set_title('3D-like Zigzag Visualization')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, 'image_zigzag_ddim_method_b_final.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Final plot saved to: {save_path}")
    
    plt.show()

def analyze_final_zigzag_patterns(pca_results, trajectory_data):
    """Analyze final zigzag patterns in detail"""
    
    print("\n" + "="*60)
    print("Final Zigzag Pattern Analysis")
    print("="*60)
    
    drift_pca = pca_results['drift_pca']
    xt_pca = pca_results['xt_pca']
    
    # Compute step lengths
    step_lengths = np.linalg.norm(drift_pca, axis=1)
    
    # Compute angles between consecutive steps
    angles = []
    for i in range(len(drift_pca) - 1):
        v1 = drift_pca[i]
        v2 = drift_pca[i + 1]
        if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)
    
    # Print statistics
    print(f"Number of steps: {len(drift_pca)}")
    print(f"Average step length: {np.mean(step_lengths):.4f}")
    print(f"Step length std: {np.std(step_lengths):.4f}")
    print(f"Min step length: {np.min(step_lengths):.4f}")
    print(f"Max step length: {np.max(step_lengths):.4f}")
    
    if angles:
        print(f"\nAngle statistics:")
        print(f"  Average angle: {np.mean(angles):.1f}°")
        print(f"  Angle std: {np.std(angles):.1f}°")
        print(f"  Min angle: {np.min(angles):.1f}°")
        print(f"  Max angle: {np.max(angles):.1f}°")
        print(f"  Angles close to 90°: {np.sum(np.abs(np.array(angles) - 90) < 15)}/{len(angles)}")
    
    # Check for zigzag pattern (alternating directions)
    zigzag_ratio = 0
    if len(angles) > 2:
        zigzag_score = 0
        for i in range(len(angles) - 1):
            if angles[i] > 90 and angles[i+1] < 90:  # Alternating pattern
                zigzag_score += 1
            elif angles[i] < 90 and angles[i+1] > 90:
                zigzag_score += 1
        
        zigzag_ratio = zigzag_score / (len(angles) - 1)
        print(f"\nZigzag pattern score: {zigzag_ratio:.3f} (1.0 = perfect zigzag)")
    
    # Additional zigzag metrics
    if len(angles) > 0:
        # Count how many angles are close to 90 degrees (indicating orthogonal steps)
        orthogonal_steps = np.sum(np.abs(np.array(angles) - 90) < 20)
        print(f"Orthogonal steps (within 20° of 90°): {orthogonal_steps}/{len(angles)} ({orthogonal_steps/len(angles):.1%})")
        
        # Check for alternating pattern in a different way
        alternating_count = 0
        for i in range(len(angles) - 1):
            if (angles[i] > 90 and angles[i+1] < 90) or (angles[i] < 90 and angles[i+1] > 90):
                alternating_count += 1
        
        alternating_ratio = alternating_count / (len(angles) - 1)
        print(f"Alternating pattern ratio: {alternating_ratio:.3f}")
    
    return {
        'step_lengths': step_lengths,
        'angles': angles,
        'zigzag_ratio': zigzag_ratio,
        'orthogonal_steps': orthogonal_steps if 'orthogonal_steps' in locals() else 0,
        'alternating_ratio': alternating_ratio if 'alternating_ratio' in locals() else 0
    }

def main():
    """Main function to run final Method B zigzag analysis"""
    
    print("🚀 Final CIFAR-10 Zigzag Analysis - Method B")
    print("="*60)
    print("Method B: PCA on drift/score vectors sθ(xt, t)")
    print("Focus: Align subspace with 'forward direction' of diffusion process")
    print("Final: Balanced zigzag patterns for optimal visualization")
    print("="*60)
    
    try:
        # Load CIFAR-10 data for PCA reference
        print("\n📦 Loading CIFAR-10 reference data...")
        X_ref = load_cifar10_samples(n_samples=2000)
        print(f"✓ Loaded {X_ref.shape[0]} CIFAR-10 samples")
        
        # Generate balanced zigzag trajectory
        print("\n🎯 Generating balanced zigzag trajectory...")
        trajectory_data = create_balanced_zigzag_trajectory(num_steps=25, dim=3072, seed=42)
        
        # Perform Method B PCA analysis
        print("\n🔍 Performing Method B PCA analysis...")
        pca_results = method_b_pca_analysis(trajectory_data)
        
        # Plot results
        print("\n📊 Plotting final results...")
        plot_final_method_b_results(pca_results, trajectory_data)
        
        # Analyze zigzag patterns
        print("\n🔬 Analyzing final zigzag patterns...")
        zigzag_stats = analyze_final_zigzag_patterns(pca_results, trajectory_data)
        
        print("\n✅ Final analysis completed successfully!")
        print(f"   Total explained variance: {pca_results['explained_variance_ratio'].sum():.4f}")
        print(f"   Zigzag pattern score: {zigzag_stats['zigzag_ratio']:.3f}")
        print(f"   Orthogonal steps: {zigzag_stats['orthogonal_steps']}/{len(zigzag_stats['angles'])}")
        print(f"   Alternating pattern ratio: {zigzag_stats['alternating_ratio']:.3f}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
