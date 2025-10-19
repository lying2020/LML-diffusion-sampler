#!/usr/bin/env python3
"""
Fixed Comprehensive Algorithm Comparison Visualization

This script creates properly sized visualizations for algorithm comparison results.
Fixed the excessive chart dimensions that were causing display issues.
"""

import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from datetime import datetime

# Set matplotlib backend and style
plt.switch_backend('Agg')
plt.style.use('default')

sys.path.append(os.getcwd())
import project as project

hessian_cg_results_dir = os.path.join(project.output_dir, "hessian_cg")
os.makedirs(hessian_cg_results_dir, exist_ok=True)

def create_comprehensive_comparison_chart(results, save_path=os.path.join(hessian_cg_results_dir, 'comprehensive_algorithm_comparison_fixed.png')):
    """Create comprehensive comparison chart with proper dimensions"""

    # Create figure with reasonable size
    fig = plt.figure(figsize=(16, 12))  # Reduced from (24, 18)
    fig.suptitle('Comprehensive Algorithm Comparison Analysis', fontsize=20, fontweight='bold', y=0.95)

    # Extract data
    algorithms = []
    times = []
    qualities = []
    fids = []
    efficiencies = []
    stabilities = []
    memories = []

    for alg_name, data in results.items():
        if 'error' not in data:
            algorithms.append(alg_name)
            times.append(data['avg_time_per_image'])
            qualities.append(data.get('avg_variance', 0))
            fids.append(data['fid_score'])
            efficiencies.append(data['efficiency'])
            stabilities.append(data['time_stability'])

            # Handle memory usage - use average if it's a list
            memory_data = data.get('memory_usage', 0)
            if isinstance(memory_data, list):
                memories.append(np.mean(memory_data))
            else:
                memories.append(memory_data)

    # Create subplots with better layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3, top=0.9, bottom=0.1, left=0.1, right=0.95)

    # 1. Generation Speed Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(range(len(algorithms)), times, color='skyblue', alpha=0.7)
    ax1.set_title('Generation Speed Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Algorithms')
    ax1.set_ylabel('Time per Image (s)')
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')

    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. Image Quality Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(range(len(algorithms)), qualities, color='lightgreen', alpha=0.7)
    ax2.set_title('Image Quality Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Algorithms')
    ax2.set_ylabel('Average Variance')
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.0f}', ha='center', va='bottom', fontsize=9)

    # 3. FID Score Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.bar(range(len(algorithms)), fids, color='lightcoral', alpha=0.7)
    ax3.set_title('FID Score Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Algorithms')
    ax3.set_ylabel('FID Score')
    ax3.set_xticks(range(len(algorithms)))
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')

    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # 4. Efficiency Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    bars4 = ax4.bar(range(len(algorithms)), efficiencies, color='gold', alpha=0.7)
    ax4.set_title('Efficiency Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Algorithms')
    ax4.set_ylabel('Efficiency (Quality/s)')
    ax4.set_xticks(range(len(algorithms)))
    ax4.set_xticklabels(algorithms, rotation=45, ha='right')

    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.0f}', ha='center', va='bottom', fontsize=9)

    # 5. Time Stability Comparison
    ax5 = fig.add_subplot(gs[2, 0])
    bars5 = ax5.bar(range(len(algorithms)), stabilities, color='plum', alpha=0.7)
    ax5.set_title('Time Stability Comparison', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Algorithms')
    ax5.set_ylabel('Time Stability')
    ax5.set_xticks(range(len(algorithms)))
    ax5.set_xticklabels(algorithms, rotation=45, ha='right')

    for i, bar in enumerate(bars5):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 6. Memory Usage Comparison
    ax6 = fig.add_subplot(gs[2, 1])
    bars6 = ax6.bar(range(len(algorithms)), memories, color='lightblue', alpha=0.7)
    ax6.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Algorithms')
    ax6.set_ylabel('Memory Usage (MB)')
    ax6.set_xticks(range(len(algorithms)))
    ax6.set_xticklabels(algorithms, rotation=45, ha='right')

    for i, bar in enumerate(bars6):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Save the plot with proper settings
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Comprehensive comparison visualization saved to: {save_path}")
    print(f"Chart dimensions: {fig.get_size_inches()}")

    return fig

def create_detailed_analysis_chart(results, save_path=os.path.join(hessian_cg_results_dir, 'detailed_algorithm_analysis_fixed.png')):
    """Create detailed analysis chart with proper dimensions"""

    # Create figure with reasonable size
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))  # Reduced from larger size
    fig.suptitle('Detailed Algorithm Analysis', fontsize=18, fontweight='bold', y=0.95)

    # Extract data
    algorithms = []
    times = []
    qualities = []
    fids = []
    efficiencies = []

    for alg_name, data in results.items():
        if 'error' not in data:
            algorithms.append(alg_name)
            times.append(data['avg_time_per_image'])
            qualities.append(data.get('avg_variance', 0))
            fids.append(data['fid_score'])
            efficiencies.append(data['efficiency'])

    # 1. Quality vs Efficiency Trade-off
    ax1.scatter(efficiencies, qualities, s=100, alpha=0.7, c=range(len(algorithms)), cmap='viridis')
    ax1.set_xlabel('Efficiency (Quality/s)')
    ax1.set_ylabel('Image Quality (Variance)')
    ax1.set_title('Quality vs Efficiency Trade-off', fontsize=14, fontweight='bold')

    # Add algorithm labels
    for i, alg in enumerate(algorithms):
        ax1.annotate(alg, (efficiencies[i], qualities[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)

    # 2. Speed vs Quality
    ax2.scatter(times, qualities, s=100, alpha=0.7, c=range(len(algorithms)), cmap='plasma')
    ax2.set_xlabel('Generation Time (s)')
    ax2.set_ylabel('Image Quality (Variance)')
    ax2.set_title('Speed vs Quality', fontsize=14, fontweight='bold')

    for i, alg in enumerate(algorithms):
        ax2.annotate(alg, (times[i], qualities[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)

    # 3. FID vs Time
    ax3.scatter(times, fids, s=100, alpha=0.7, c=range(len(algorithms)), cmap='coolwarm')
    ax3.set_xlabel('Generation Time (s)')
    ax3.set_ylabel('FID Score')
    ax3.set_title('FID Score vs Generation Time', fontsize=14, fontweight='bold')

    for i, alg in enumerate(algorithms):
        ax3.annotate(alg, (times[i], fids[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)

    # 4. Efficiency vs FID
    ax4.scatter(efficiencies, fids, s=100, alpha=0.7, c=range(len(algorithms)), cmap='inferno')
    ax4.set_xlabel('Efficiency (Quality/s)')
    ax4.set_ylabel('FID Score')
    ax4.set_title('Efficiency vs FID Score', fontsize=14, fontweight='bold')

    for i, alg in enumerate(algorithms):
        ax4.annotate(alg, (efficiencies[i], fids[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)

    # Save with proper settings
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Detailed analysis chart saved to: {save_path}")
    print(f"Chart dimensions: {fig.get_size_inches()}")

    return fig

def main():
    parser = argparse.ArgumentParser(description="Fixed Comprehensive Algorithm Comparison Visualization")
    parser.add_argument('--results_file', type=str,
                        default=os.path.join(hessian_cg_results_dir, 'comprehensive_algorithm_comparison_fixed.json'),
                        help='Path to the results JSON file')
    parser.add_argument('--output_dir', type=str, default=hessian_cg_results_dir,
                        help='Output directory for visualizations')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    print(f"Loading results from: {args.results_file}")
    with open(args.results_file, 'r') as f:
        results = json.load(f)

    print(f"Loaded results for {len(results)} algorithms")

    # Create visualizations
    print("\nCreating comprehensive comparison chart...")
    fig1 = create_comprehensive_comparison_chart(
        results,
        os.path.join(args.output_dir, 'comprehensive_algorithm_comparison_fixed.png')
    )
    plt.close(fig1)

    print("\nCreating detailed analysis chart...")
    fig2 = create_detailed_analysis_chart(
        results,
        os.path.join(args.output_dir, 'detailed_algorithm_analysis_fixed.png')
    )
    plt.close(fig2)

    print("\n✅ All visualizations created successfully!")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main()
