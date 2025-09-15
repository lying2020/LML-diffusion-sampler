#!/usr/bin/env python3
"""
Visualization script for Hessian-Free vs LML analysis

This script creates comprehensive visualizations of the analysis results
including performance charts, scalability plots, and convergence curves.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List
import argparse
import os
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HessianAnalysisVisualizer:
    """Visualizer for Hessian analysis results"""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.results = self.load_results()
        
    def load_results(self) -> Dict:
        """Load results from JSON file"""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def plot_performance_comparison(self, save_path: str = None):
        """Plot performance comparison between methods"""
        
        perf_results = self.results.get('performance', {})
        if not perf_results:
            print("No performance data available for visualization")
            return
        
        # Extract data
        methods = []
        times = []
        qualities = []
        memories = []
        efficiencies = []
        
        for method, result in perf_results.items():
            if 'error' not in result:
                methods.append(method)
                times.append(result['avg_time_per_image'])
                qualities.append(result['avg_quality'])
                memories.append(result['avg_memory_usage'])
                efficiencies.append(result['avg_quality'] / result['avg_time_per_image'])
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hessian-Free vs LML Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Generation Time
        colors = ['red' if 'lml' in m else 'blue' for m in methods]
        bars1 = ax1.bar(range(len(methods)), times, color=colors, alpha=0.7)
        ax1.set_title('Generation Time per Image')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars1, times)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{time:.3f}s', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Image Quality
        bars2 = ax2.bar(range(len(methods)), qualities, color=colors, alpha=0.7)
        ax2.set_title('Image Quality Score')
        ax2.set_ylabel('Quality Score')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        
        for i, (bar, quality) in enumerate(zip(bars2, qualities)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{quality:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Memory Usage
        bars3 = ax3.bar(range(len(methods)), memories, color=colors, alpha=0.7)
        ax3.set_title('Memory Usage')
        ax3.set_ylabel('Memory (MB)')
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels(methods, rotation=45, ha='right')
        
        for i, (bar, memory) in enumerate(zip(bars3, memories)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{memory:.1f}MB', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Efficiency (Quality per Second)
        bars4 = ax4.bar(range(len(methods)), efficiencies, color=colors, alpha=0.7)
        ax4.set_title('Efficiency (Quality per Second)')
        ax4.set_ylabel('Efficiency Score')
        ax4.set_xticks(range(len(methods)))
        ax4.set_xticklabels(methods, rotation=45, ha='right')
        
        for i, (bar, efficiency) in enumerate(zip(bars4, efficiencies)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{efficiency:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='LML Methods'),
                          Patch(facecolor='blue', alpha=0.7, label='Hessian-Free Methods')]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_scalability_analysis(self, save_path: str = None):
        """Plot scalability analysis across batch sizes"""
        
        scale_results = self.results.get('scalability', {})
        if not scale_results:
            print("No scalability data available for visualization")
            return
        
        # Extract data
        methods = list(scale_results.keys())
        batch_sizes = [1, 2, 4, 8]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Scalability Analysis: Batch Size vs Performance', fontsize=16, fontweight='bold')
        
        # Plot 1: Time vs Batch Size
        for method in methods:
            times = []
            for batch_size in batch_sizes:
                if batch_size in scale_results[method] and 'error' not in scale_results[method][batch_size]:
                    times.append(scale_results[method][batch_size]['avg_time_per_image'])
                else:
                    times.append(np.nan)
            
            if any(not np.isnan(times)):
                ax1.plot(batch_sizes, times, marker='o', label=method, linewidth=2, markersize=8)
        
        ax1.set_title('Generation Time vs Batch Size')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Time per Image (seconds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Quality vs Batch Size
        for method in methods:
            qualities = []
            for batch_size in batch_sizes:
                if batch_size in scale_results[method] and 'error' not in scale_results[method][batch_size]:
                    qualities.append(scale_results[method][batch_size]['avg_quality'])
                else:
                    qualities.append(np.nan)
            
            if any(not np.isnan(qualities)):
                ax2.plot(batch_sizes, qualities, marker='s', label=method, linewidth=2, markersize=8)
        
        ax2.set_title('Image Quality vs Batch Size')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Quality Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scalability analysis plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_convergence_analysis(self, save_path: str = None):
        """Plot convergence analysis across inference steps"""
        
        conv_results = self.results.get('convergence', {})
        if not conv_results:
            print("No convergence data available for visualization")
            return
        
        # Extract data
        methods = list(conv_results.keys())
        inference_steps = [5, 10, 15, 20, 25, 30]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Convergence Analysis: Inference Steps vs Performance', fontsize=16, fontweight='bold')
        
        # Plot 1: Time vs Inference Steps
        for method in methods:
            times = []
            for steps in inference_steps:
                if steps in conv_results[method] and 'error' not in conv_results[method][steps]:
                    times.append(conv_results[method][steps]['avg_time_per_image'])
                else:
                    times.append(np.nan)
            
            if any(not np.isnan(times)):
                ax1.plot(inference_steps, times, marker='o', label=method, linewidth=2, markersize=8)
        
        ax1.set_title('Generation Time vs Inference Steps')
        ax1.set_xlabel('Inference Steps')
        ax1.set_ylabel('Time per Image (seconds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Quality vs Inference Steps
        for method in methods:
            qualities = []
            for steps in inference_steps:
                if steps in conv_results[method] and 'error' not in conv_results[method][steps]:
                    qualities.append(conv_results[method][steps]['avg_quality'])
                else:
                    qualities.append(np.nan)
            
            if any(not np.isnan(qualities)):
                ax2.plot(inference_steps, qualities, marker='s', label=method, linewidth=2, markersize=8)
        
        ax2.set_title('Image Quality vs Inference Steps')
        ax2.set_xlabel('Inference Steps')
        ax2.set_ylabel('Quality Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence analysis plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_stability_analysis(self, save_path: str = None):
        """Plot stability analysis across different parameters"""
        
        stab_results = self.results.get('stability', {})
        if not stab_results:
            print("No stability data available for visualization")
            return
        
        # Extract data
        methods = list(stab_results.keys())
        param_names = ['low_reg', 'medium_reg', 'high_reg', 'very_high_reg']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Stability Analysis: Parameters vs Performance', fontsize=16, fontweight='bold')
        
        # Plot 1: Quality vs Parameters
        x_pos = np.arange(len(param_names))
        width = 0.35
        
        for i, method in enumerate(methods):
            qualities = []
            for param in param_names:
                if param in stab_results[method] and 'error' not in stab_results[method][param]:
                    qualities.append(stab_results[method][param]['avg_quality'])
                else:
                    qualities.append(0)
            
            if any(qualities):
                ax1.bar(x_pos + i*width, qualities, width, label=method, alpha=0.8)
        
        ax1.set_title('Image Quality vs Regularization Parameters')
        ax1.set_xlabel('Parameter Configuration')
        ax1.set_ylabel('Quality Score')
        ax1.set_xticks(x_pos + width/2)
        ax1.set_xticklabels(param_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Quality Standard Deviation vs Parameters
        for i, method in enumerate(methods):
            std_qualities = []
            for param in param_names:
                if param in stab_results[method] and 'error' not in stab_results[method][param]:
                    std_qualities.append(stab_results[method][param]['std_quality'])
                else:
                    std_qualities.append(0)
            
            if any(std_qualities):
                ax2.bar(x_pos + i*width, std_qualities, width, label=method, alpha=0.8)
        
        ax2.set_title('Quality Stability vs Regularization Parameters')
        ax2.set_xlabel('Parameter Configuration')
        ax2.set_ylabel('Quality Standard Deviation')
        ax2.set_xticks(x_pos + width/2)
        ax2.set_xticklabels(param_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Stability analysis plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_comprehensive_summary(self, save_path: str = None):
        """Plot comprehensive summary of all analyses"""
        
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('Comprehensive Hessian-Free vs LML Analysis Summary', fontsize=20, fontweight='bold')
        
        # Performance comparison (top row)
        perf_results = self.results.get('performance', {})
        if perf_results:
            methods = [k for k in perf_results.keys() if 'error' not in perf_results[k]]
            times = [perf_results[k]['avg_time_per_image'] for k in methods]
            qualities = [perf_results[k]['avg_quality'] for k in methods]
            
            # Time comparison
            ax1 = fig.add_subplot(gs[0, 0])
            colors = ['red' if 'lml' in m else 'blue' for m in methods]
            bars = ax1.bar(range(len(methods)), times, color=colors, alpha=0.7)
            ax1.set_title('Generation Time', fontweight='bold')
            ax1.set_ylabel('Time (s)')
            ax1.set_xticks(range(len(methods)))
            ax1.set_xticklabels(methods, rotation=45, ha='right')
            
            # Quality comparison
            ax2 = fig.add_subplot(gs[0, 1])
            bars = ax2.bar(range(len(methods)), qualities, color=colors, alpha=0.7)
            ax2.set_title('Image Quality', fontweight='bold')
            ax2.set_ylabel('Quality Score')
            ax2.set_xticks(range(len(methods)))
            ax2.set_xticklabels(methods, rotation=45, ha='right')
            
            # Efficiency comparison
            ax3 = fig.add_subplot(gs[0, 2])
            efficiencies = [q/t for q, t in zip(qualities, times)]
            bars = ax3.bar(range(len(methods)), efficiencies, color=colors, alpha=0.7)
            ax3.set_title('Efficiency', fontweight='bold')
            ax3.set_ylabel('Quality/Time')
            ax3.set_xticks(range(len(methods)))
            ax3.set_xticklabels(methods, rotation=45, ha='right')
        
        # Scalability analysis (middle row)
        scale_results = self.results.get('scalability', {})
        if scale_results:
            batch_sizes = [1, 2, 4, 8]
            
            # Time scalability
            ax4 = fig.add_subplot(gs[1, 0])
            for method in list(scale_results.keys())[:3]:  # Limit to 3 methods for clarity
                times = []
                for batch_size in batch_sizes:
                    if batch_size in scale_results[method] and 'error' not in scale_results[method][batch_size]:
                        times.append(scale_results[method][batch_size]['avg_time_per_image'])
                    else:
                        times.append(np.nan)
                if any(not np.isnan(times)):
                    ax4.plot(batch_sizes, times, marker='o', label=method, linewidth=2)
            ax4.set_title('Time Scalability', fontweight='bold')
            ax4.set_xlabel('Batch Size')
            ax4.set_ylabel('Time (s)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Quality scalability
            ax5 = fig.add_subplot(gs[1, 1])
            for method in list(scale_results.keys())[:3]:
                qualities = []
                for batch_size in batch_sizes:
                    if batch_size in scale_results[method] and 'error' not in scale_results[method][batch_size]:
                        qualities.append(scale_results[method][batch_size]['avg_quality'])
                    else:
                        qualities.append(np.nan)
                if any(not np.isnan(qualities)):
                    ax5.plot(batch_sizes, qualities, marker='s', label=method, linewidth=2)
            ax5.set_title('Quality Scalability', fontweight='bold')
            ax5.set_xlabel('Batch Size')
            ax5.set_ylabel('Quality Score')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Convergence analysis (bottom row)
        conv_results = self.results.get('convergence', {})
        if conv_results:
            inference_steps = [5, 10, 15, 20, 25, 30]
            
            # Time convergence
            ax7 = fig.add_subplot(gs[2, 0])
            for method in list(conv_results.keys())[:3]:
                times = []
                for steps in inference_steps:
                    if steps in conv_results[method] and 'error' not in conv_results[method][steps]:
                        times.append(conv_results[method][steps]['avg_time_per_image'])
                    else:
                        times.append(np.nan)
                if any(not np.isnan(times)):
                    ax7.plot(inference_steps, times, marker='o', label=method, linewidth=2)
            ax7.set_title('Time Convergence', fontweight='bold')
            ax7.set_xlabel('Inference Steps')
            ax7.set_ylabel('Time (s)')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            
            # Quality convergence
            ax8 = fig.add_subplot(gs[2, 1])
            for method in list(conv_results.keys())[:3]:
                qualities = []
                for steps in inference_steps:
                    if steps in conv_results[method] and 'error' not in conv_results[method][steps]:
                        qualities.append(conv_results[method][steps]['avg_quality'])
                    else:
                        qualities.append(np.nan)
                if any(not np.isnan(qualities)):
                    ax8.plot(inference_steps, qualities, marker='s', label=method, linewidth=2)
            ax8.set_title('Quality Convergence', fontweight='bold')
            ax8.set_xlabel('Inference Steps')
            ax8.set_ylabel('Quality Score')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # Summary statistics (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        if perf_results:
            # Calculate summary statistics
            lml_methods = [k for k in perf_results.keys() if 'lml' in k and 'error' not in perf_results[k]]
            hf_methods = [k for k in perf_results.keys() if 'hessian_free' in k and 'error' not in perf_results[k]]
            
            if lml_methods and hf_methods:
                avg_lml_time = np.mean([perf_results[k]['avg_time_per_image'] for k in lml_methods])
                avg_hf_time = np.mean([perf_results[k]['avg_time_per_image'] for k in hf_methods])
                avg_lml_quality = np.mean([perf_results[k]['avg_quality'] for k in lml_methods])
                avg_hf_quality = np.mean([perf_results[k]['avg_quality'] for k in hf_methods])
                
                summary_text = f"""
SUMMARY STATISTICS

LML Methods:
  Avg Time: {avg_lml_time:.3f}s
  Avg Quality: {avg_lml_quality:.0f}

Hessian-Free Methods:
  Avg Time: {avg_hf_time:.3f}s
  Avg Quality: {avg_hf_quality:.0f}

Performance Difference:
  Time: {((avg_hf_time - avg_lml_time) / avg_lml_time * 100):+.1f}%
  Quality: {((avg_hf_quality - avg_lml_quality) / avg_lml_quality * 100):+.1f}%
                """
                
                ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive summary plot saved to: {save_path}")
        else:
            plt.show()
    
    def generate_all_plots(self, output_dir: str = "hessian_analysis_plots"):
        """Generate all plots and save them"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating comprehensive Hessian analysis visualizations...")
        
        # Generate individual plots
        self.plot_performance_comparison(os.path.join(output_dir, "performance_comparison.png"))
        self.plot_scalability_analysis(os.path.join(output_dir, "scalability_analysis.png"))
        self.plot_convergence_analysis(os.path.join(output_dir, "convergence_analysis.png"))
        self.plot_stability_analysis(os.path.join(output_dir, "stability_analysis.png"))
        self.plot_comprehensive_summary(os.path.join(output_dir, "comprehensive_summary.png"))
        
        print(f"\nAll plots saved to: {output_dir}/")
        print("Generated plots:")
        print("  - performance_comparison.png: Performance metrics comparison")
        print("  - scalability_analysis.png: Batch size scalability analysis")
        print("  - convergence_analysis.png: Inference steps convergence analysis")
        print("  - stability_analysis.png: Parameter stability analysis")
        print("  - comprehensive_summary.png: Complete analysis summary")

def main():
    parser = argparse.ArgumentParser(description="Visualize Hessian analysis results")
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to the JSON results file')
    parser.add_argument('--output_dir', type=str, default='hessian_analysis_plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file {args.results_file} not found")
        return
    
    # Create visualizer and generate plots
    visualizer = HessianAnalysisVisualizer(args.results_file)
    visualizer.generate_all_plots(args.output_dir)

if __name__ == '__main__':
    main()
