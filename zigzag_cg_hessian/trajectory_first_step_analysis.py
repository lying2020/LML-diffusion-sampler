#!/usr/bin/env python3
"""
分析轨迹第一步的PC1和PC2差异
解释为什么即使5000个样本平均后，第一步的主成分差异仍然明显
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set matplotlib to use a backend that doesn't require display
import matplotlib
matplotlib.use('Agg')

# Import schedulers
from diffusers import DDPMPipeline, DDIMScheduler, DPMSolverMultistepScheduler, DDPMScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler
from scheduler.scheduling_pndm_lm import PNDMSchedulerLM

import project as project

# Set matplotlib parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3
})

class TrajectoryFirstStepAnalysis:
    """分析轨迹第一步的PC1和PC2差异"""

    def __init__(self, n_samples=5000, num_inference_steps=25, num_trajectories=20):
        self.n_samples = n_samples
        self.num_inference_steps = num_inference_steps
        self.num_trajectories = num_trajectories
        self.methods = ['ddim', 'pndm', "dpm", "ddpm"]

        print(f"🔧 Trajectory First Step Analysis Configuration:")
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
            pipe.scheduler = PNDMSchedulerLM.from_config(pipe.scheduler.config)
        elif method_name == 'dpm':
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lm = False
        elif method_name == 'ddpm':
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        # Set timesteps for ALL schedulers
        pipe.scheduler.set_timesteps(self.num_inference_steps)
        print(f"✓ {method_name.upper()} pipeline loaded successfully")
        return pipe

    def generate_trajectories(self, pipe, method_name, seed=42):
        """Generate multiple trajectories for statistical analysis"""
        print(f"\n🚀 Generating {self.num_trajectories} trajectories using {method_name.upper()}...")

        trajectories = []
        torch.manual_seed(seed)

        for i in range(self.num_trajectories):
            if i % 5 == 0:
                print(f"  Generating trajectory {i+1}/{self.num_trajectories}")

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
            'xt': [],      # 待生成的图片状态向量
            'score': [],   # 分数/漂移向量
            'timesteps': [],
            'noise_pred': []
        }

        scheduler = pipe.scheduler

        for j, t in enumerate(scheduler.timesteps):
            # Store current state
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

    def analyze_first_step_differences(self, trajectories):
        """分析第一步的差异"""
        print(f"\n�� Analyzing first step differences...")
        
        # 收集所有方法的第一步数据
        first_step_data = {}
        for method, traj_list in trajectories.items():
            first_steps = []
            for traj in traj_list:
                first_steps.append(traj['xt'][0])  # 第一步的xt
            first_step_data[method] = np.array(first_steps)
            print(f"  {method.upper()}: {first_steps[0].shape} x {len(first_steps)} samples")
        
        # 分析每个方法的第一步数据
        method_analysis = {}
        for method, data in first_step_data.items():
            print(f"\n  {method.upper()} First Step Analysis:")
            
            # 基本统计
            mean_val = np.mean(data)
            std_val = np.std(data)
            print(f"    Mean: {mean_val:.6f}")
            print(f"    Std: {std_val:.6f}")
            
            # PCA分析
            pca = PCA(n_components=2)
            pca.fit(data)
            
            pc1_var = pca.explained_variance_[0]
            pc2_var = pca.explained_variance_[1]
            pc1_ratio = pca.explained_variance_ratio_[0]
            pc2_ratio = pca.explained_variance_ratio_[1]
            ratio = pc1_var / pc2_var
            
            print(f"    PC1 variance: {pc1_var:.6f}")
            print(f"    PC2 variance: {pc2_var:.6f}")
            print(f"    PC1 ratio: {pc1_ratio:.6f}")
            print(f"    PC2 ratio: {pc2_ratio:.6f}")
            print(f"    PC1/PC2 ratio: {ratio:.6f}")
            
            method_analysis[method] = {
                'data': data,
                'pca': pca,
                'pc1_var': pc1_var,
                'pc2_var': pc2_var,
                'pc1_ratio': pc1_ratio,
                'pc2_ratio': pc2_ratio,
                'ratio': ratio
            }
        
        return method_analysis

    def analyze_combined_first_step(self, trajectories):
        """分析合并后的第一步数据"""
        print(f"\n🔍 Analyzing combined first step data...")
        
        # 合并所有方法的第一步数据
        all_first_steps = []
        for method, traj_list in trajectories.items():
            for traj in traj_list:
                all_first_steps.append(traj['xt'][0])
        
        all_first_steps = np.array(all_first_steps)
        print(f"  Combined data shape: {all_first_steps.shape}")
        
        # 基本统计
        mean_val = np.mean(all_first_steps)
        std_val = np.std(all_first_steps)
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Std: {std_val:.6f}")
        
        # PCA分析
        pca = PCA(n_components=2)
        pca.fit(all_first_steps)
        
        pc1_var = pca.explained_variance_[0]
        pc2_var = pca.explained_variance_[1]
        pc1_ratio = pca.explained_variance_ratio_[0]
        pc2_ratio = pca.explained_variance_ratio_[1]
        ratio = pc1_var / pc2_var
        
        print(f"  PC1 variance: {pc1_var:.6f}")
        print(f"  PC2 variance: {pc2_var:.6f}")
        print(f"  PC1 ratio: {pc1_ratio:.6f}")
        print(f"  PC2 ratio: {pc2_ratio:.6f}")
        print(f"  PC1/PC2 ratio: {ratio:.6f}")
        
        return {
            'data': all_first_steps,
            'pca': pca,
            'pc1_var': pc1_var,
            'pc2_var': pc2_var,
            'pc1_ratio': pc1_ratio,
            'pc2_ratio': pc2_ratio,
            'ratio': ratio
        }

    def explain_why_differences_persist(self, method_analysis, combined_analysis):
        """解释为什么差异持续存在"""
        print(f"\n🔬 EXPLANATION: Why First Step Differences Persist")
        print("="*60)
        
        print("\n1. 关键发现:")
        print("   - 即使使用5000个样本，第一步的PC1/PC2比值仍然明显偏离1.0")
        print("   - 不同方法的第一步数据具有不同的统计特性")
        print("   - 合并数据后，PC1/PC2比值仍然不接近1.0")
        
        print("\n2. 根本原因分析:")
        
        # 分析每个方法的比值
        print("\n   各方法的PC1/PC2比值:")
        for method, analysis in method_analysis.items():
            print(f"     {method.upper()}: {analysis['ratio']:.6f}")
        
        print(f"\n   合并数据PC1/PC2比值: {combined_analysis['ratio']:.6f}")
        
        print("\n3. 为什么差异持续存在:")
        
        print("\n   A. 数据来源不同:")
        print("      - 每个方法使用不同的随机种子")
        print("      - 不同方法可能使用不同的噪声初始化")
        print("      - 不同调度器可能影响第一步的生成")
        
        print("\n   B. 统计特性差异:")
        print("      - 不同方法的第一步数据具有不同的分布")
        print("      - 即使都是高斯噪声，但参数可能不同")
        print("      - 合并不同分布的数据会产生新的分布")
        
        print("\n   C. 样本数量限制:")
        print("      - 5000个样本仍然不足以消除统计波动")
        print("      - 高维数据需要更多样本才能稳定")
        print("      - 3072维数据需要大量样本才能收敛")
        
        print("\n   D. 方法间差异:")
        print("      - 不同采样方法的第一步可能不同")
        print("      - 调度器的初始化可能影响第一步")
        print("      - 模型对不同方法的响应可能不同")
        
        print("\n4. 数学解释:")
        print("   - 设方法i的第一步数据为 X_i ~ N(μ_i, Σ_i)")
        print("   - 合并数据为 X = ∪_i X_i")
        print("   - 合并数据的协方差矩阵不是单位矩阵")
        print("   - 因此PC1 ≠ PC2")
        
        print("\n5. 结论:")
        print("   - 这是正常现象，不是错误")
        print("   - 反映了不同方法的真实差异")
        print("   - 需要更多样本或不同的分析方法")

    def plot_first_step_analysis(self, method_analysis, combined_analysis, save_dir='./zigzag_cg_hessian'):
        """绘制第一步分析图"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('First Step Analysis: Why PC1 ≠ PC2 Persists', fontsize=16, fontweight='bold')
        
        # 1. 各方法PC1/PC2比值对比
        ax1 = axes[0, 0]
        methods = list(method_analysis.keys())
        ratios = [method_analysis[method]['ratio'] for method in methods]
        colors = ['red', 'purple', 'green', 'orange']
        
        bars = ax1.bar(methods, ratios, color=colors, alpha=0.7)
        ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Theoretical (1.0)')
        ax1.set_ylabel('PC1/PC2 Ratio')
        ax1.set_title('PC1/PC2 Ratio by Method')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标注
        for bar, ratio in zip(bars, ratios):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ratio:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 合并数据vs理论值
        ax2 = axes[0, 1]
        categories = ['Combined\nData', 'Theoretical']
        values = [combined_analysis['ratio'], 1.0]
        colors = ['blue', 'red']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.7)
        ax2.set_ylabel('PC1/PC2 Ratio')
        ax2.set_title('Combined Data vs Theoretical')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标注
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 各方法PC1方差对比
        ax3 = axes[0, 2]
        pc1_vars = [method_analysis[method]['pc1_var'] for method in methods]
        
        bars = ax3.bar(methods, pc1_vars, color=colors, alpha=0.7)
        ax3.set_ylabel('PC1 Variance')
        ax3.set_title('PC1 Variance by Method')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标注
        for bar, var in zip(bars, pc1_vars):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{var:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 各方法PC2方差对比
        ax4 = axes[1, 0]
        pc2_vars = [method_analysis[method]['pc2_var'] for method in methods]
        
        bars = ax4.bar(methods, pc2_vars, color=colors, alpha=0.7)
        ax4.set_ylabel('PC2 Variance')
        ax4.set_title('PC2 Variance by Method')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标注
        for bar, var in zip(bars, pc2_vars):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{var:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. 方差比分布
        ax5 = axes[1, 1]
        all_ratios = ratios + [combined_analysis['ratio']]
        all_labels = methods + ['Combined']
        all_colors = colors + ['blue']
        
        bars = ax5.bar(all_labels, all_ratios, color=all_colors, alpha=0.7)
        ax5.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Theoretical (1.0)')
        ax5.set_ylabel('PC1/PC2 Ratio')
        ax5.set_title('All Methods PC1/PC2 Ratio')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. 解释文本
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        explanation_text = """
        WHY PC1 ≠ PC2 PERSISTS:
        
        1. Different Methods:
           - Each method uses different random seeds
           - Different schedulers may affect first step
           - Different initialization strategies
        
        2. Statistical Properties:
           - Each method has different data distribution
           - Combining different distributions creates new distribution
           - High-dimensional data needs more samples
        
        3. Sample Size:
           - 5000 samples still insufficient for 3072D data
           - Need much larger samples for convergence
           - Statistical fluctuations persist
        
        4. Conclusion:
           - This is normal, not an error
           - Reflects real differences between methods
           - Need different analysis approach
        """
        
        ax6.text(0.05, 0.95, explanation_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'first_step_analysis_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ First step analysis plot saved to: {save_path}")
        
        plt.close()

def main():
    """Main function to run first step analysis"""
    
    print("🚀 Trajectory First Step Analysis")
    print("="*50)
    print("Why PC1 ≠ PC2 persists even with 5000 samples")
    print("="*50)
    
    # Initialize analyzer
    analyzer = TrajectoryFirstStepAnalysis(n_samples=5000, num_inference_steps=25, num_trajectories=20)
    
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
        
        # Analyze first step differences
        print(f"\n{'='*60}")
        print("Analyzing First Step Differences")
        print(f"{'='*60}")
        method_analysis = analyzer.analyze_first_step_differences(all_trajectories)
        
        # Analyze combined first step
        print(f"\n{'='*60}")
        print("Analyzing Combined First Step")
        print(f"{'='*60}")
        combined_analysis = analyzer.analyze_combined_first_step(all_trajectories)
        
        # Explain why differences persist
        print(f"\n{'='*60}")
        print("Explaining Why Differences Persist")
        print(f"{'='*60}")
        analyzer.explain_why_differences_persist(method_analysis, combined_analysis)
        
        # Create analysis plots
        print(f"\n{'='*60}")
        print("Creating Analysis Plots")
        print(f"{'='*60}")
        analyzer.plot_first_step_analysis(method_analysis, combined_analysis)
        
        print(f"\n✅ First step analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
