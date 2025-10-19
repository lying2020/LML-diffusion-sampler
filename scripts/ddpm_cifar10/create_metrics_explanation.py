#!/usr/bin/env python3
"""
Create Metrics Explanation Chart - Final Version

This script creates a comprehensive chart explaining all evaluation metrics
with proper font handling and clear explanations.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

def create_metrics_explanation_chart():
    """Create a comprehensive chart explaining all metrics"""

    # Set style and font
    plt.style.use('default')
    sns.set_palette("Set2")

    # Set font to avoid encoding issues
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10

    # Create figure
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Algorithm Evaluation Metrics Definition', fontsize=24, fontweight='bold', y=0.95)

    # 1. Quality Metrics
    ax1 = plt.subplot(3, 3, 1)
    ax1.set_title('Quality Metrics', fontsize=16, fontweight='bold', pad=20)
    ax1.axis('off')

    quality_text = """
    Primary Metric: Image Variance

    Definition:
    variance = mean([var(img) for img in images])

    Sub-metrics:
    • Sharpness: Laplacian variance
    • Contrast: RMS contrast
    • Color Diversity: RGB channel differences

    Range: 0 → ∞ (higher is better)

    Physical Meaning:
    Higher variance indicates richer
    image content and better quality
    """

    ax1.text(0.1, 0.9, quality_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # 2. Efficiency Metrics
    ax2 = plt.subplot(3, 3, 2)
    ax2.set_title('Efficiency Metrics', fontsize=16, fontweight='bold', pad=20)
    ax2.axis('off')

    efficiency_text = """
    Primary Metric: Quality per Second

    Definition:
    efficiency = avg_variance / avg_time_per_image

    Unit: quality/s (quality per second)

    Physical Meaning:
    Measures how much quality can be
    generated per unit time

    Example:
    Hessian-Free-Fast = 289,663 quality/s
    means 289,663 quality units
    generated per second

    Range: 0 → ∞ (higher is better)
    """

    ax2.text(0.1, 0.9, efficiency_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 3. Stability Metrics
    ax3 = plt.subplot(3, 3, 3)
    ax3.set_title('Stability Metrics', fontsize=16, fontweight='bold', pad=20)
    ax3.axis('off')

    stability_text = """
    Primary Metric: Time Stability

    Definition:
    time_stability = 1.0 / (1.0 + std_time / mean_time)

    Range: 0-1 (higher is better)

    Physical Meaning:
    Measures consistency of generation
    time across different samples

    Interpretation:
    • 0.9-1.0: Very stable
    • 0.7-0.9: Stable
    • 0.5-0.7: Moderate
    • 0.0-0.5: Unstable

    Quality Stability:
    Similar formula for quality consistency
    """

    ax3.text(0.1, 0.9, stability_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # 4. FID Score
    ax4 = plt.subplot(3, 3, 4)
    ax4.set_title('FID Score', fontsize=16, fontweight='bold', pad=20)
    ax4.axis('off')

    fid_text = """
    Definition: Fréchet Inception Distance

    Formula:
    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r Σ_g))

    Where:
    • μ_r, μ_g: Mean features (real, generated)
    • Σ_r, Σ_g: Covariance matrices
    • Tr: Trace of matrix

    Range: 0 → ∞ (lower is better)

    Interpretation:
    • 0-10: Excellent
    • 10-50: Good
    • 50-100: Fair
    • 100+: Poor

    Note: Requires real dataset for
    accurate computation
    """

    ax4.text(0.1, 0.9, fid_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 5. Speed Metrics
    ax5 = plt.subplot(3, 3, 5)
    ax5.set_title('Speed Metrics', fontsize=16, fontweight='bold', pad=20)
    ax5.axis('off')

    speed_text = """
    Primary Metric: Time per Image

    Definition:
    avg_time_per_image = total_time / num_images

    Sub-metrics:
    • Min time: Fastest generation
    • Max time: Slowest generation
    • Std time: Time variability

    Unit: seconds per image

    Range: 0 → ∞ (lower is better)

    Typical Values:
    • Real-time: < 0.1s
    • Fast: 0.1-0.5s
    • Moderate: 0.5-2.0s
    • Slow: > 2.0s

    Speed Improvement:
    ((slowest - fastest) / slowest) × 100%
    """

    ax5.text(0.1, 0.9, speed_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # 6. Memory Metrics
    ax6 = plt.subplot(3, 3, 6)
    ax6.set_title('Memory Metrics', fontsize=16, fontweight='bold', pad=20)
    ax6.axis('off')

    memory_text = """
    Primary Metric: Memory Usage

    Definition:
    avg_memory_usage = mean([mem_after - mem_before])

    Unit: MB (megabytes)

    Range: 0 → ∞ (lower is better)

    Measurement:
    • mem_before: Memory before generation
    • mem_after: Memory after generation
    • Peak memory: Maximum memory used

    Typical Values:
    • Low: < 100 MB
    • Moderate: 100-500 MB
    • High: 500-1000 MB
    • Very High: > 1000 MB

    Memory Efficiency:
    Quality per MB of memory used
    """

    ax6.text(0.1, 0.9, memory_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))

    # 7. Trade-off Analysis
    ax7 = plt.subplot(3, 3, 7)
    ax7.set_title('Trade-off Analysis', fontsize=16, fontweight='bold', pad=20)
    ax7.axis('off')

    tradeoff_text = """
    Key Trade-offs:

    1. Speed vs Quality:
    • Faster algorithms may sacrifice quality
    • Quality algorithms may be slower
    • Optimal point depends on use case

    2. Memory vs Performance:
    • More memory can improve performance
    • Memory constraints limit algorithms
    • Hessian-Free balances both

    3. Accuracy vs Speed:
    • Explicit methods: High accuracy, slow
    • Approximate methods: Lower accuracy, fast
    • Hessian-Free: Good balance

    4. Stability vs Performance:
    • Stable algorithms may be slower
    • Fast algorithms may be less stable
    • Choose based on requirements
    """

    ax7.text(0.1, 0.9, tradeoff_text, transform=ax7.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # 8. Algorithm Comparison
    ax8 = plt.subplot(3, 3, 8)
    ax8.set_title('Algorithm Comparison', fontsize=16, fontweight='bold', pad=20)
    ax8.axis('off')

    comparison_text = """
    Algorithm Characteristics:

    Baseline Methods (DDIM, DPM, etc.):
    • Speed: Fast
    • Quality: Moderate
    • Memory: Low
    • Stability: High

    LML Methods:
    • Speed: Moderate
    • Quality: High
    • Memory: Low
    • Stability: High

    Explicit Hessian:
    • Speed: Slow
    • Quality: Very High
    • Memory: Very High
    • Stability: Moderate

    Hessian-Free:
    • Speed: Fast
    • Quality: High
    • Memory: Low
    • Stability: High
    """

    ax8.text(0.1, 0.9, comparison_text, transform=ax8.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.8))

    # 9. References and Citations
    ax9 = plt.subplot(3, 3, 9)
    ax9.set_title('References and Citations', fontsize=16, fontweight='bold', pad=20)
    ax9.axis('off')

    references_text = """
    Key References:

    FID Score:
    Heusel et al., "GANs Trained by a Two Time-Scale
    Update Rule Converge to a Local Nash Equilibrium",
    NeurIPS 2017

    Diffusion Models:
    Ho et al., "Denoising Diffusion Probabilistic Models",
    NeurIPS 2020

    DDIM:
    Song et al., "Denoising Diffusion Implicit Models",
    ICLR 2021

    DPM-Solver:
    Lu et al., "DPM-Solver: A Fast ODE Solver for
    Diffusion Probabilistic Model Sampling",
    NeurIPS 2022

    LML:
    Original LML paper (if available)

    Hessian-Free:
    Pearlmutter, "Fast Exact Multiplication by the
    Hessian", Neural Computation 1994
    """

    ax9.text(0.1, 0.9, references_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgoldenrodyellow', alpha=0.8))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)

    # Save the plot
    save_path = 'output/test/metrics_definition_explanation.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics explanation chart saved to: {save_path}")

    return fig

if __name__ == '__main__':
    create_metrics_explanation_chart()
