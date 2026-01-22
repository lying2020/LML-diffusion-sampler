#!/usr/bin/env python3
"""
ICLR Paper Format: Zigzag Phenomenon Visualization
Visualizing oscillating trajectory converging to center with decreasing amplitude
Similar to the conceptual diagram with concentric ellipses and red oscillating path
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Set matplotlib to use a backend that doesn't require display
import matplotlib
matplotlib.use('Agg')

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'results')
pngs_zigzag_dir = os.path.join(results_dir, "zigzag")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(pngs_zigzag_dir, exist_ok=True)

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

class ICLRZigzagVisualizer:
    """ICLR paper format visualizer for zigzag phenomenon"""

    def __init__(self, num_points=80):
        self.num_points = num_points
        print(f"🔧 ICLR Zigzag Visualization Configuration:")
        print(f"   - Number of trajectory points: {self.num_points}")
        print(f"   - Visualization: Oscillating path with decreasing amplitude")

    def generate_zigzag_trajectory(self):
        """Generate a synthetic zigzag trajectory with decreasing amplitude"""
        print(f"\n🚀 Generating synthetic zigzag trajectory...")

        # Parameters for zigzag path (more compact)
        num_points = self.num_points
        center = np.array([0.0, 0.0])  # Center point
        start_radius = 4.5  # Starting distance from center (further reduced for compactness)
        end_radius = 0.15  # Ending distance from center

        # Generate trajectory points
        trajectory = []

        # Starting position (left side, along horizontal axis - ellipse's major axis)
        start_x = -start_radius
        start_y = 0.0  # Start at center vertically

        # Create a path that oscillates toward center with decreasing amplitude
        # Main direction: horizontal (along ellipse's major axis)
        # Oscillation: vertical (perpendicular to major axis)
        for i in range(num_points):
            # Progress from start to end (0 to 1)
            t = i / (num_points - 1)

            # Main direction: from left toward center (horizontal)
            # Linear interpolation from start to center along x-axis
            main_x = start_x * (1 - t) + center[0] * t
            main_y = center[1]  # Keep y at center (main path is horizontal)

            # Oscillation: vertical (perpendicular to horizontal main direction)
            # Create sharp zigzag pattern with alternating direction
            # Number of oscillations increases as we approach center
            num_oscillations = 1.125 + 5.625 * t  # More oscillations near center (frequency reduced by 1/4, i.e., 0.75x)
            oscillation_phase = num_oscillations * t * 2 * np.pi

            # Oscillation amplitude decreases exponentially (but more slowly to keep it visible longer)
            max_oscillation_amp = 1.8  # Further reduced for more compact visualization
            oscillation_amp = max_oscillation_amp * np.exp(-1.5 * t)  # Slower decay to maintain amplitude longer

            # Create sharp zigzag using sawtooth-like pattern
            # Use triangular wave for sharp zigzag effect with larger range
            phase_normalized = (oscillation_phase % (2*np.pi)) / (2*np.pi)
            if phase_normalized < 0.5:
                zigzag_value = 2 * phase_normalized - 0.5  # Rising edge: -0.5 to 0.5
            else:
                zigzag_value = 1.5 - 2 * phase_normalized  # Falling edge: 0.5 to -0.5

            # Amplify the zigzag effect for more dramatic oscillations
            zigzag_value = zigzag_value * 1.2  # Increase range by 20%

            # Adjust oscillation to match ellipse shape (horizontal wider, vertical narrower)
            # Ellipse aspect ratio: b/a = 0.45 (vertical/horizontal)
            ellipse_aspect_ratio = 0.45  # b/a ratio

            # Increase oscillation amplitude to better fill the elliptical space
            # Oscillation is vertical (y-direction), perpendicular to horizontal main path
            adjusted_amp = oscillation_amp * 1.4  # Increase amplitude to better fit ellipse

            # Position: main path (horizontal) + vertical oscillation (zigzag)
            x = main_x  # Horizontal movement
            y = main_y + adjusted_amp * zigzag_value  # Vertical oscillation

            trajectory.append([x, y])

        trajectory = np.array(trajectory)
        print(f"✓ Generated zigzag trajectory with {len(trajectory)} points")
        return trajectory

    def plot_iclr_zigzag_visualization(self, trajectory_pca):
        """Plot ICLR paper format zigzag visualization with concentric ellipses"""

        # Calculate trajectory center and extent for ellipse placement
        center = np.array([0.0, 0.0])  # Use origin as center
        max_dist = np.max(np.linalg.norm(trajectory_pca - center, axis=1))

        # Create figure - single plot optimized for ICLR paper format (more compact, narrower width)
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        # fig.suptitle('Zigzag Phenomenon in Diffusion Sampling\n(Oscillating Path with Decreasing Amplitude)',
        #              fontsize=18, fontweight='bold', y=0.95)

        # Draw concentric ellipses (representing energy levels or potential contours)
        n_ellipses = 7
        ellipse_colors = plt.cm.Greys(np.linspace(0.3, 0.7, n_ellipses))

        # Calculate trajectory extent in x and y directions separately
        max_x = np.max(np.abs(trajectory_pca[:, 0]))
        max_y = np.max(np.abs(trajectory_pca[:, 1]))
        # Use the larger extent to determine ellipse size, maintaining aspect ratio
        max_extent = max(max_x, max_y)
        ellipse_aspect_ratio = 0.45  # b/a ratio (vertical/horizontal)

        for i in range(n_ellipses):
            # Create ellipse with aspect ratio similar to the image (more elliptical and compact)
            # Maintain 45% aspect ratio while fitting trajectory
            scale = (1.0 - i * 0.12)
            a = max_extent * 1.15 * scale  # Major axis (horizontal)
            b = a * ellipse_aspect_ratio  # Minor axis (vertical), 45% of major

            # Create ellipse points
            theta = np.linspace(0, 2*np.pi, 100)
            ellipse_x = center[0] + a * np.cos(theta)
            ellipse_y = center[1] + b * np.sin(theta)

            ax.plot(ellipse_x, ellipse_y, color=ellipse_colors[i],
                   linewidth=1.5, alpha=0.6, linestyle='-')

        # Plot trajectory with oscillating pattern
        # Create color gradient from start to end
        n_points = len(trajectory_pca)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))

        # Plot trajectory segments with varying linewidth to show amplitude decrease
        for i in range(n_points - 1):
            # Calculate distance from center to show amplitude
            dist_from_center = np.linalg.norm(trajectory_pca[i] - center)
            # Linewidth decreases as we approach center (decreasing amplitude)
            linewidth = 4.0 * (dist_from_center / max_dist) + 1.0

            ax.plot([trajectory_pca[i, 0], trajectory_pca[i+1, 0]],
                   [trajectory_pca[i, 1], trajectory_pca[i+1, 1]],
                   color='#E74C3C',  # Red color like in the image
                   linewidth=linewidth, alpha=0.9, zorder=4)

        # Mark start point (green dot)
        start_point = trajectory_pca[0]
        ax.scatter(start_point[0], start_point[1],
                  c='#27AE60', s=250, marker='o', label='Start', zorder=5,
                  edgecolors='black', linewidth=1.5)

        # Mark end point (orange dot)
        end_point = trajectory_pca[-1]
        ax.scatter(end_point[0], end_point[1],
                  c='#E67E22', s=250, marker='s', label='End', zorder=5,
                  edgecolors='black', linewidth=1.5)

        # Add step markers along the trajectory
        step_interval = max(1, n_points // 15)
        for i in range(0, n_points, step_interval):
            dist_from_center = np.linalg.norm(trajectory_pca[i] - center)
            marker_size = 80 * (dist_from_center / max_dist) + 20  # Reduced marker size
            ax.scatter(trajectory_pca[i, 0], trajectory_pca[i, 1],
                      c='#E74C3C', s=marker_size, marker='o', alpha=0.6, zorder=3,
                      edgecolors='white', linewidth=0.8)

        # Set labels and formatting
        # ax.set_title('Zigzag Trajectory: Oscillation with Decreasing Amplitude',
        #             fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=18, loc='upper right', ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.set_aspect('equal', adjustable='box')

        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Remove borders (spines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Adjust layout for ICLR paper format (more compact)
        plt.tight_layout(rect=[0, 0, 1, 1])  # Full use of space

        # Save the plot with high DPI for ICLR paper
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(pngs_zigzag_dir, f'iclr_zigzag_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\n✓ ICLR format zigzag visualization plot saved to: {save_path}")

        plt.show()

def main():
    """Main function to run ICLR paper format zigzag visualization"""

    print("🚀 ICLR Paper Format: Zigzag Phenomenon Visualization")
    print("="*80)
    print("Visualizing oscillating trajectory converging to center with decreasing amplitude")
    print("="*80)

    # Initialize visualizer
    visualizer = ICLRZigzagVisualizer(num_points=80)

    try:
        # Generate synthetic zigzag trajectory
        print(f"\n{'='*60}")
        print("Generating Zigzag Trajectory")
        print(f"{'='*60}")
        trajectory_pca = visualizer.generate_zigzag_trajectory()

        # Create ICLR paper format zigzag visualization
        print(f"\n{'='*60}")
        print("Creating ICLR Paper Format Zigzag Visualization...")
        print(f"{'='*60}")
        visualizer.plot_iclr_zigzag_visualization(trajectory_pca)

        print(f"\n✅ ICLR paper format zigzag visualization completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
