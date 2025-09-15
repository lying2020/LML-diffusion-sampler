#!/usr/bin/env python3
"""
Visualize Comprehensive Algorithm Comparison Results

This script creates comprehensive visualization charts for the algorithm comparison results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import os

def load_results(filename):
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def create_comprehensive_visualization(results, save_path='comprehensive_algorithm_comparison.png'):
    """Create comprehensive visualization of algorithm comparison results"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Comprehensive Algorithm Comparison Analysis', fontsize=20, fontweight='bold', y=0.98)
    
    # Extract data
    algorithms = []
    times = []
    qualities = []
    fids = []
    efficiencies = []
    stabilities = []
    
    for alg_name, data in results.items():
        if 'error' not in data:
            algorithms.append(alg_name)
            times.append(data['avg_time_per_image'])
            qualities.append(data.get('avg_variance', 0))
            fids.append(data['fid_score'])
            efficiencies.append(data['efficiency'])
            stabilities.append(data['time_stability'])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Algorithm': algorithms,
        'Time (s)': times,
        'Quality': qualities,
        'FID': fids,
        'Efficiency': efficiencies,
        'Stability': stabilities
    })
    
    # 1. Performance Overview (2x2 grid)
    ax1 = plt.subplot(3, 3, 1)
    bars1 = ax1.bar(range(len(algorithms)), times, color='skyblue', alpha=0.7)
    ax1.set_title('Generation Speed Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time per Image (s)')
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Quality Comparison
    ax2 = plt.subplot(3, 3, 2)
    bars2 = ax2.bar(range(len(algorithms)), qualities, color='lightgreen', alpha=0.7)
    ax2.set_title('Image Quality Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Quality (Variance)')
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 3. FID Score Comparison (log scale)
    ax3 = plt.subplot(3, 3, 3)
    bars3 = ax3.bar(range(len(algorithms)), fids, color='salmon', alpha=0.7)
    ax3.set_title('FID Score Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('FID Score (log scale)')
    ax3.set_yscale('log')
    ax3.set_xticks(range(len(algorithms)))
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Efficiency Comparison
    ax4 = plt.subplot(3, 3, 4)
    bars4 = ax4.bar(range(len(algorithms)), efficiencies, color='gold', alpha=0.7)
    ax4.set_title('Efficiency Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Efficiency (Quality/s)')
    ax4.set_xticks(range(len(algorithms)))
    ax4.set_xticklabels(algorithms, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 5. Stability Comparison
    ax5 = plt.subplot(3, 3, 5)
    bars5 = ax5.bar(range(len(algorithms)), stabilities, color='plum', alpha=0.7)
    ax5.set_title('Time Stability Comparison', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Stability Score')
    ax5.set_xticks(range(len(algorithms)))
    ax5.set_xticklabels(algorithms, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars5):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 6. Speed vs Quality Scatter Plot
    ax6 = plt.subplot(3, 3, 6)
    scatter = ax6.scatter(times, qualities, s=100, alpha=0.7, c=efficiencies, cmap='viridis')
    ax6.set_title('Speed vs Quality Trade-off', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Time per Image (s)')
    ax6.set_ylabel('Quality (Variance)')
    ax6.grid(True, alpha=0.3)
    
    # Add algorithm labels
    for i, alg in enumerate(algorithms):
        ax6.annotate(alg, (times[i], qualities[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Efficiency (Quality/s)')
    
    # 7. Algorithm Category Comparison
    ax7 = plt.subplot(3, 3, 7)
    
    # Categorize algorithms
    baseline_algs = [alg for alg in algorithms if alg in ['DDIM', 'DPM-Solver']]
    lml_algs = [alg for alg in algorithms if 'LML' in alg or 'DDIM-LM' in alg]
    hf_algs = [alg for alg in algorithms if 'Hessian-Free' in alg]
    
    categories = ['Baseline', 'LML', 'Hessian-Free']
    avg_times = []
    avg_qualities = []
    
    for cat_algs in [baseline_algs, lml_algs, hf_algs]:
        if cat_algs:
            cat_times = [times[algorithms.index(alg)] for alg in cat_algs]
            cat_qualities = [qualities[algorithms.index(alg)] for alg in cat_algs]
            avg_times.append(np.mean(cat_times))
            avg_qualities.append(np.mean(cat_qualities))
        else:
            avg_times.append(0)
            avg_qualities.append(0)
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax7.bar(x - width/2, avg_times, width, label='Avg Time (s)', alpha=0.7)
    ax7_twin = ax7.twinx()
    bars2 = ax7_twin.bar(x + width/2, avg_qualities, width, label='Avg Quality', alpha=0.7, color='orange')
    
    ax7.set_title('Algorithm Category Comparison', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Algorithm Category')
    ax7.set_ylabel('Average Time (s)', color='blue')
    ax7_twin.set_ylabel('Average Quality', color='orange')
    ax7.set_xticks(x)
    ax7.set_xticklabels(categories)
    ax7.grid(True, alpha=0.3)
    
    # 8. Performance Radar Chart
    ax8 = plt.subplot(3, 3, 8, projection='polar')
    
    # Normalize metrics for radar chart
    normalized_times = 1 - (np.array(times) - np.min(times)) / (np.max(times) - np.min(times))
    normalized_qualities = (np.array(qualities) - np.min(qualities)) / (np.max(qualities) - np.min(qualities))
    normalized_efficiencies = (np.array(efficiencies) - np.min(efficiencies)) / (np.max(efficiencies) - np.min(efficiencies))
    normalized_stabilities = np.array(stabilities)
    
    # Select top 3 algorithms for radar chart
    top_3_indices = np.argsort(efficiencies)[-3:]
    
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, idx in enumerate(top_3_indices):
        values = [
            normalized_times[idx],
            normalized_qualities[idx],
            normalized_efficiencies[idx],
            normalized_stabilities[idx]
        ]
        values += values[:1]  # Complete the circle
        
        ax8.plot(angles, values, 'o-', linewidth=2, label=algorithms[idx])
        ax8.fill(angles, values, alpha=0.25)
    
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(['Speed', 'Quality', 'Efficiency', 'Stability'])
    ax8.set_title('Top 3 Algorithms Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax8.grid(True)
    
    # 9. Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate summary statistics
    fastest_alg = algorithms[np.argmin(times)]
    best_quality_alg = algorithms[np.argmax(qualities)]
    most_efficient_alg = algorithms[np.argmax(efficiencies)]
    most_stable_alg = algorithms[np.argmax(stabilities)]
    
    summary_text = f"""
    🏆 PERFORMANCE SUMMARY
    
    ⚡ Fastest: {fastest_alg}
       Time: {min(times):.3f}s per image
    
    🎨 Best Quality: {best_quality_alg}
       Quality: {max(qualities):.0f}
    
    ⚖️ Most Efficient: {most_efficient_alg}
       Efficiency: {max(efficiencies):.0f} quality/s
    
    🔒 Most Stable: {most_stable_alg}
       Stability: {max(stabilities):.3f}
    
    📊 Speed Improvement: {((max(times) - min(times)) / max(times) * 100):.1f}%
    
    🎯 Total Algorithms: {len(algorithms)}
    """
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive comparison visualization saved to: {save_path}")
    
    return fig

def create_detailed_analysis_chart(results, save_path='detailed_algorithm_analysis.png'):
    """Create detailed analysis chart focusing on key metrics"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Algorithm Analysis', fontsize=18, fontweight='bold')
    
    # Extract data
    algorithms = []
    times = []
    qualities = []
    fids = []
    efficiencies = []
    stabilities = []
    
    for alg_name, data in results.items():
        if 'error' not in data:
            algorithms.append(alg_name)
            times.append(data['avg_time_per_image'])
            qualities.append(data.get('avg_variance', 0))
            fids.append(data['fid_score'])
            efficiencies.append(data['efficiency'])
            stabilities.append(data['time_stability'])
    
    # 1. Performance Heatmap
    ax1.set_title('Performance Heatmap', fontsize=14, fontweight='bold')
    
    # Normalize data for heatmap
    metrics = ['Time', 'Quality', 'FID', 'Efficiency', 'Stability']
    data_matrix = np.array([times, qualities, fids, efficiencies, stabilities])
    
    # Normalize each metric
    normalized_data = np.zeros_like(data_matrix)
    for i in range(len(metrics)):
        if i in [0, 2]:  # Time and FID (lower is better)
            normalized_data[i] = 1 - (data_matrix[i] - np.min(data_matrix[i])) / (np.max(data_matrix[i]) - np.min(data_matrix[i]))
        else:  # Quality, Efficiency, Stability (higher is better)
            normalized_data[i] = (data_matrix[i] - np.min(data_matrix[i])) / (np.max(data_matrix[i]) - np.min(data_matrix[i]))
    
    im = ax1.imshow(normalized_data, cmap='RdYlGn', aspect='auto')
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.set_yticks(range(len(metrics)))
    ax1.set_yticklabels(metrics)
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(algorithms)):
            text = ax1.text(j, i, f'{normalized_data[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax1, shrink=0.8)
    
    # 2. Efficiency vs Quality Scatter
    ax2.scatter(qualities, efficiencies, s=150, alpha=0.7, c=times, cmap='viridis')
    ax2.set_title('Quality vs Efficiency Trade-off', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Quality (Variance)')
    ax2.set_ylabel('Efficiency (Quality/s)')
    ax2.grid(True, alpha=0.3)
    
    # Add algorithm labels
    for i, alg in enumerate(algorithms):
        ax2.annotate(alg, (qualities[i], efficiencies[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, alpha=0.8)
    
    # Add colorbar
    scatter = ax2.scatter(qualities, efficiencies, s=150, alpha=0.7, c=times, cmap='viridis')
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Time per Image (s)')
    
    # 3. Speed Distribution
    ax3.hist(times, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('Speed Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time per Image (s)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Add mean line
    mean_time = np.mean(times)
    ax3.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.3f}s')
    ax3.legend()
    
    # 4. Algorithm Ranking
    ax4.set_title('Algorithm Ranking by Efficiency', fontsize=14, fontweight='bold')
    
    # Sort by efficiency
    sorted_indices = np.argsort(efficiencies)[::-1]
    sorted_algorithms = [algorithms[i] for i in sorted_indices]
    sorted_efficiencies = [efficiencies[i] for i in sorted_indices]
    
    bars = ax4.barh(range(len(sorted_algorithms)), sorted_efficiencies, color='lightcoral', alpha=0.7)
    ax4.set_yticks(range(len(sorted_algorithms)))
    ax4.set_yticklabels(sorted_algorithms)
    ax4.set_xlabel('Efficiency (Quality/s)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.0f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Detailed analysis chart saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Comprehensive Algorithm Comparison")
    parser.add_argument('--results_file', type=str, 
                        default='comprehensive_algorithm_comparison_20250915_032407.json',
                        help='Path to the results JSON file')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for visualization files')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_file)
    
    # Create visualizations
    comprehensive_fig = create_comprehensive_visualization(
        results, 
        os.path.join(args.output_dir, 'comprehensive_algorithm_comparison.png')
    )
    
    detailed_fig = create_detailed_analysis_chart(
        results,
        os.path.join(args.output_dir, 'detailed_algorithm_analysis.png')
    )
    
    print("✅ Visualization complete!")
    print("Generated files:")
    print("  - comprehensive_algorithm_comparison.png")
    print("  - detailed_algorithm_analysis.png")

if __name__ == '__main__':
    main()
