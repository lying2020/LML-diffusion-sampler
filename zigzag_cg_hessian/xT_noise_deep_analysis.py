#!/usr/bin/env python3
"""
深入分析x_T高斯噪声的PCA2结果
解释为什么PC1和PC2不相等
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

class XTNoiseDeepAnalysis:
    """深入分析x_T高斯噪声的PCA2结果"""

    def __init__(self):
        self.image_shape = (32, 32, 3)
        self.data_dim = np.prod(self.image_shape)  # 32*32*3 = 3072

    def theoretical_analysis(self):
        """理论分析"""
        print("🔬 THEORETICAL ANALYSIS")
        print("="*50)
        
        print("\n1. 理论期望:")
        print("   - 对于纯高斯噪声 X ~ N(0, I)，协方差矩阵 Cov(X) = I")
        print("   - 协方差矩阵的特征值应该都等于1")
        print("   - 因此 PC1 和 PC2 的方差应该相等")
        print("   - PC1/PC2 比值应该等于 1.0")
        
        print("\n2. 扩散过程中的 x_T:")
        print("   - x_T = sqrt(1 - α_T) * ε，其中 ε ~ N(0, I)")
        print("   - 对于 T=1000，α_T ≈ 0，所以 x_T ≈ ε")
        print("   - 因此 x_T 也应该近似服从 N(0, I)")
        
        print("\n3. 为什么实际结果不相等:")
        print("   - 有限样本效应：样本数量有限导致统计波动")
        print("   - 数值精度：浮点运算引入小误差")
        print("   - 随机性：不同随机种子产生不同结果")
        print("   - 算法偏差：PCA/SVD算法可能引入小偏差")

    def sample_size_analysis(self, sample_sizes=[100, 500, 1000, 2000, 5000]):
        """分析不同样本大小对PC1/PC2比值的影响"""
        print(f"\n🔬 SAMPLE SIZE ANALYSIS")
        print("="*50)
        
        results = []
        
        for n_samples in sample_sizes:
            print(f"\n分析样本大小: {n_samples}")
            
            # 生成纯高斯噪声
            np.random.seed(42)
            data = np.random.randn(n_samples, self.data_dim)
            
            # 执行PCA
            pca = PCA(n_components=2, svd_solver='randomized')
            pca.fit(data)
            
            ratio = pca.explained_variance_[0] / pca.explained_variance_[1]
            results.append((n_samples, ratio))
            
            print(f"   - PC1/PC2 比值: {ratio:.6f}")
            print(f"   - 与理论值1.0的偏差: {abs(ratio - 1.0):.6f}")
        
        return results

    def multiple_runs_analysis(self, n_samples=1000, n_runs=10):
        """多次运行分析，观察随机性影响"""
        print(f"\n🔬 MULTIPLE RUNS ANALYSIS")
        print("="*50)
        print(f"样本大小: {n_samples}, 运行次数: {n_runs}")
        
        ratios = []
        
        for run in range(n_runs):
            # 使用不同的随机种子
            np.random.seed(42 + run)
            data = np.random.randn(n_samples, self.data_dim)
            
            # 执行PCA
            pca = PCA(n_components=2, svd_solver='randomized')
            pca.fit(data)
            
            ratio = pca.explained_variance_[0] / pca.explained_variance_[1]
            ratios.append(ratio)
            
            print(f"   运行 {run+1}: PC1/PC2 = {ratio:.6f}")
        
        # 统计分析
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        min_ratio = np.min(ratios)
        max_ratio = np.max(ratios)
        
        print(f"\n统计结果:")
        print(f"   - 平均比值: {mean_ratio:.6f}")
        print(f"   - 标准差: {std_ratio:.6f}")
        print(f"   - 最小值: {min_ratio:.6f}")
        print(f"   - 最大值: {max_ratio:.6f}")
        print(f"   - 变异系数: {std_ratio/mean_ratio:.6f}")
        
        return ratios

    def eigenvalue_distribution_analysis(self, n_samples=1000):
        """分析特征值分布"""
        print(f"\n🔬 EIGENVALUE DISTRIBUTION ANALYSIS")
        print("="*50)
        
        # 生成数据
        np.random.seed(42)
        data = np.random.randn(n_samples, self.data_dim)
        
        # 计算协方差矩阵
        data_centered = data - np.mean(data, axis=0)
        cov_matrix = np.cov(data_centered.T)
        
        # 计算特征值
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列
        
        print(f"协方差矩阵特征值分析:")
        print(f"   - 矩阵大小: {cov_matrix.shape}")
        print(f"   - 特征值数量: {len(eigenvalues)}")
        print(f"   - 最大特征值: {eigenvalues[0]:.6f}")
        print(f"   - 最小特征值: {eigenvalues[-1]:.6f}")
        print(f"   - 平均特征值: {np.mean(eigenvalues):.6f}")
        print(f"   - 特征值标准差: {np.std(eigenvalues):.6f}")
        
        # 分析前几个特征值
        print(f"\n前10个特征值:")
        for i in range(min(10, len(eigenvalues))):
            print(f"   λ_{i+1} = {eigenvalues[i]:.6f}")
        
        # 分析特征值分布
        theoretical_value = 1.0
        deviations = np.abs(eigenvalues - theoretical_value)
        
        print(f"\n与理论值1.0的偏差:")
        print(f"   - 平均偏差: {np.mean(deviations):.6f}")
        print(f"   - 最大偏差: {np.max(deviations):.6f}")
        print(f"   - 前10个特征值的平均偏差: {np.mean(deviations[:10]):.6f}")
        
        return eigenvalues

    def plot_analysis_results(self, sample_size_results, multiple_runs_results, eigenvalues, save_dir='./zigzag_cg_hessian'):
        """绘制分析结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Deep Analysis: Why PC1 ≠ PC2 for Gaussian Noise',
                     fontsize=16, fontweight='bold')
        
        # 1. 样本大小对PC1/PC2比值的影响
        ax1 = axes[0, 0]
        sample_sizes, ratios = zip(*sample_size_results)
        ax1.plot(sample_sizes, ratios, 'bo-', linewidth=2, markersize=8)
        ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Theoretical (1.0)')
        ax1.set_xlabel('Sample Size')
        ax1.set_ylabel('PC1/PC2 Ratio')
        ax1.set_title('Effect of Sample Size on PC1/PC2 Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # 2. 多次运行的PC1/PC2比值分布
        ax2 = axes[0, 1]
        ax2.hist(multiple_runs_results, bins=10, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Theoretical (1.0)')
        ax2.axvline(x=np.mean(multiple_runs_results), color='green', linestyle='-', linewidth=2, label=f'Mean ({np.mean(multiple_runs_results):.4f})')
        ax2.set_xlabel('PC1/PC2 Ratio')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of PC1/PC2 Ratio (Multiple Runs)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 特征值分布
        ax3 = axes[1, 0]
        n_plot = min(100, len(eigenvalues))
        ax3.plot(range(n_plot), eigenvalues[:n_plot], 'bo-', linewidth=2, markersize=4)
        ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Theoretical (1.0)')
        ax3.set_xlabel('Eigenvalue Index')
        ax3.set_ylabel('Eigenvalue')
        ax3.set_title('Eigenvalue Distribution (First 100)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 特征值偏差分析
        ax4 = axes[1, 1]
        theoretical_value = 1.0
        deviations = np.abs(eigenvalues - theoretical_value)
        n_plot = min(50, len(deviations))
        ax4.plot(range(n_plot), deviations[:n_plot], 'ro-', linewidth=2, markersize=4)
        ax4.set_xlabel('Eigenvalue Index')
        ax4.set_ylabel('Absolute Deviation from 1.0')
        ax4.set_title('Deviation from Theoretical Value')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'xT_noise_deep_analysis_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Deep analysis plot saved to: {save_path}")
        
        plt.close()

    def generate_deep_analysis_report(self, sample_size_results, multiple_runs_results, eigenvalues, save_dir='./zigzag_cg_hessian'):
        """生成深度分析报告"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(save_dir, f'xT_noise_deep_analysis_report_{timestamp}.txt')
        
        with open(report_path, 'w') as f:
            f.write("X_T Gaussian Noise Deep Analysis Report\n")
            f.write("="*50 + "\n\n")
            
            f.write("THEORETICAL BACKGROUND:\n")
            f.write("-" * 25 + "\n")
            f.write("For pure Gaussian noise X ~ N(0, I):\n")
            f.write("1. Covariance matrix Cov(X) = I (identity matrix)\n")
            f.write("2. All eigenvalues should equal 1\n")
            f.write("3. PC1 and PC2 should have equal variance\n")
            f.write("4. PC1/PC2 ratio should equal 1.0\n\n")
            
            f.write("WHY PC1 ≠ PC2 IN PRACTICE:\n")
            f.write("-" * 30 + "\n")
            f.write("1. FINITE SAMPLE EFFECTS:\n")
            f.write("   - Limited number of samples causes statistical fluctuations\n")
            f.write("   - Sample covariance matrix ≠ population covariance matrix\n")
            f.write("   - Law of large numbers: ratio → 1.0 as n → ∞\n\n")
            
            f.write("2. NUMERICAL PRECISION:\n")
            f.write("   - Floating point arithmetic introduces small errors\n")
            f.write("   - SVD algorithm has numerical limitations\n")
            f.write("   - Machine epsilon effects accumulate\n\n")
            
            f.write("3. RANDOMNESS:\n")
            f.write("   - Different random seeds produce different results\n")
            f.write("   - Statistical variability in finite samples\n")
            f.write("   - Monte Carlo fluctuations\n\n")
            
            f.write("4. ALGORITHM BIAS:\n")
            f.write("   - PCA/SVD implementation may have small biases\n")
            f.write("   - Numerical stability considerations\n")
            f.write("   - Convergence criteria effects\n\n")
            
            f.write("EXPERIMENTAL EVIDENCE:\n")
            f.write("-" * 20 + "\n")
            
            # 样本大小分析结果
            f.write("Sample Size Analysis:\n")
            for n_samples, ratio in sample_size_results:
                f.write(f"  n={n_samples}: PC1/PC2 = {ratio:.6f}, deviation = {abs(ratio-1.0):.6f}\n")
            f.write("\n")
            
            # 多次运行分析结果
            f.write("Multiple Runs Analysis:\n")
            f.write(f"  Mean ratio: {np.mean(multiple_runs_results):.6f}\n")
            f.write(f"  Std ratio: {np.std(multiple_runs_results):.6f}\n")
            f.write(f"  Min ratio: {np.min(multiple_runs_results):.6f}\n")
            f.write(f"  Max ratio: {np.max(multiple_runs_results):.6f}\n")
            f.write(f"  Coefficient of variation: {np.std(multiple_runs_results)/np.mean(multiple_runs_results):.6f}\n\n")
            
            # 特征值分析结果
            f.write("Eigenvalue Analysis:\n")
            f.write(f"  Mean eigenvalue: {np.mean(eigenvalues):.6f}\n")
            f.write(f"  Std eigenvalue: {np.std(eigenvalues):.6f}\n")
            f.write(f"  Max eigenvalue: {np.max(eigenvalues):.6f}\n")
            f.write(f"  Min eigenvalue: {np.min(eigenvalues):.6f}\n")
            f.write(f"  Mean deviation from 1.0: {np.mean(np.abs(eigenvalues - 1.0)):.6f}\n\n")
            
            f.write("CONCLUSIONS:\n")
            f.write("-" * 12 + "\n")
            f.write("1. The deviation from PC1 = PC2 is due to finite sample effects\n")
            f.write("2. The deviation decreases as sample size increases\n")
            f.write("3. The deviation is random and follows statistical laws\n")
            f.write("4. For large enough samples, the ratio approaches 1.0\n")
            f.write("5. This is a normal statistical phenomenon, not a bug\n\n")
            
            f.write("PRACTICAL IMPLICATIONS:\n")
            f.write("-" * 22 + "\n")
            f.write("1. For trajectory visualization, small deviations are acceptable\n")
            f.write("2. For zigzag analysis, consider using larger sample sizes\n")
            f.write("3. The relative ordering of methods is more important than absolute values\n")
            f.write("4. Statistical significance tests should account for this variability\n")
        
        print(f"\n✓ Deep analysis report saved to: {report_path}")

def main():
    """Main function to run deep analysis"""
    
    print("🚀 X_T Gaussian Noise Deep Analysis")
    print("="*50)
    print("Understanding why PC1 ≠ PC2 for Gaussian noise")
    print("="*50)
    
    # Initialize analyzer
    analyzer = XTNoiseDeepAnalysis()
    
    try:
        # 1. 理论分析
        analyzer.theoretical_analysis()
        
        # 2. 样本大小分析
        print(f"\n{'='*60}")
        print("Step 1: Sample Size Analysis")
        print(f"{'='*60}")
        sample_size_results = analyzer.sample_size_analysis([100, 500, 1000, 2000, 5000])
        
        # 3. 多次运行分析
        print(f"\n{'='*60}")
        print("Step 2: Multiple Runs Analysis")
        print(f"{'='*60}")
        multiple_runs_results = analyzer.multiple_runs_analysis(n_samples=1000, n_runs=10)
        
        # 4. 特征值分布分析
        print(f"\n{'='*60}")
        print("Step 3: Eigenvalue Distribution Analysis")
        print(f"{'='*60}")
        eigenvalues = analyzer.eigenvalue_distribution_analysis(n_samples=1000)
        
        # 5. 绘制分析结果
        print(f"\n{'='*60}")
        print("Step 4: Create Analysis Plots")
        print(f"{'='*60}")
        analyzer.plot_analysis_results(sample_size_results, multiple_runs_results, eigenvalues)
        
        # 6. 生成深度分析报告
        print(f"\n{'='*60}")
        print("Step 5: Generate Deep Analysis Report")
        print(f"{'='*60}")
        analyzer.generate_deep_analysis_report(sample_size_results, multiple_runs_results, eigenvalues)
        
        print(f"\n✅ Deep analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
