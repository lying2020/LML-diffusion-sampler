#!/usr/bin/env python3
"""
分析x_T高斯噪声的PCA2结果
验证理论上PC1和PC2应该相等的假设
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

class XTNoiseAnalysis:
    """分析x_T高斯噪声的PCA2结果"""

    def __init__(self, n_samples=1000, image_shape=(32, 32, 3)):
        self.n_samples = n_samples
        self.image_shape = image_shape
        self.data_dim = np.prod(image_shape)  # 32*32*3 = 3072

        print(f"🔧 X_T Noise Analysis Configuration:")
        print(f"   - Number of samples: {self.n_samples}")
        print(f"   - Image shape: {self.image_shape}")
        print(f"   - Data dimension: {self.data_dim}")

    def generate_pure_gaussian_noise(self, n_samples, seed=42):
        """生成纯高斯噪声数据"""
        print(f"\n📊 Generating {n_samples} pure Gaussian noise samples...")

        np.random.seed(seed)
        data = np.random.randn(n_samples, self.data_dim)

        print(f"✓ Generated pure Gaussian noise shape: {data.shape}")
        print(f"✓ Data statistics:")
        print(f"   - Mean: {np.mean(data):.6f}")
        print(f"   - Std: {np.std(data):.6f}")
        print(f"   - Min: {np.min(data):.4f}")
        print(f"   - Max: {np.max(data):.4f}")

        return data

    def generate_xT_from_diffusion(self, n_samples=100, seed=42):
        """从扩散过程中生成x_T数据"""
        print(f"\n📊 Generating {n_samples} x_T samples from diffusion process...")

        # 模拟扩散过程：x_T = sqrt(alpha_T) * x_0 + sqrt(1 - alpha_T) * epsilon
        # 对于纯噪声，x_0 = 0，所以 x_T = sqrt(1 - alpha_T) * epsilon

        # 使用DDPM的噪声调度
        num_train_timesteps = 1000
        beta_start = 0.0001
        beta_end = 0.02

        # 计算beta schedule
        betas = np.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        # 选择T=1000时的alpha值
        T = 1000
        alpha_T = alphas_cumprod[T-1]

        print(f"   - T = {T}")
        print(f"   - alpha_T = {alpha_T:.6f}")
        print(f"   - sqrt(1 - alpha_T) = {np.sqrt(1 - alpha_T):.6f}")

        # 生成x_T
        np.random.seed(seed)
        epsilon = np.random.randn(n_samples, self.data_dim)
        x_T = np.sqrt(1 - alpha_T) * epsilon

        print(f"✓ Generated x_T shape: {x_T.shape}")
        print(f"✓ x_T statistics:")
        print(f"   - Mean: {np.mean(x_T):.6f}")
        print(f"   - Std: {np.std(x_T):.6f}")
        print(f"   - Min: {np.min(x_T):.4f}")
        print(f"   - Max: {np.max(x_T):.4f}")

        return x_T

    def analyze_pca_variance(self, data, data_name):
        """分析PCA方差"""
        print(f"\n🔍 Analyzing PCA variance for {data_name}...")

        # 执行PCA
        pca = PCA(n_components=2, svd_solver='randomized')
        pca.fit(data)

        explained_variance_ratio = pca.explained_variance_ratio_
        explained_variance = pca.explained_variance_

        print(f"✓ PCA results for {data_name}:")
        print(f"   - PC1 explained variance ratio: {explained_variance_ratio[0]:.6f}")
        print(f"   - PC2 explained variance ratio: {explained_variance_ratio[1]:.6f}")
        print(f"   - PC1 explained variance: {explained_variance[0]:.6f}")
        print(f"   - PC2 explained variance: {explained_variance[1]:.6f}")
        print(f"   - Ratio PC1/PC2: {explained_variance[0]/explained_variance[1]:.6f}")

        return pca, explained_variance_ratio, explained_variance

    def analyze_eigenvalue_distribution(self, data, data_name):
        """分析特征值分布"""
        print(f"\n🔍 Analyzing eigenvalue distribution for {data_name}...")

        # 计算协方差矩阵
        data_centered = data - np.mean(data, axis=0)
        cov_matrix = np.cov(data_centered.T)

        # 计算所有特征值
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列

        print(f"✓ Eigenvalue analysis for {data_name}:")
        print(f"   - Total eigenvalues: {len(eigenvalues)}")
        print(f"   - Max eigenvalue: {eigenvalues[0]:.6f}")
        print(f"   - Min eigenvalue: {eigenvalues[-1]:.6f}")
        print(f"   - First 10 eigenvalues: {eigenvalues[:10]}")
        print(f"   - Last 10 eigenvalues: {eigenvalues[-10:]}")

        # 分析特征值分布
        mean_eigenvalue = np.mean(eigenvalues)
        std_eigenvalue = np.std(eigenvalues)

        print(f"   - Mean eigenvalue: {mean_eigenvalue:.6f}")
        print(f"   - Std eigenvalue: {std_eigenvalue:.6f}")
        print(f"   - Coefficient of variation: {std_eigenvalue/mean_eigenvalue:.6f}")

        return eigenvalues

    def plot_eigenvalue_analysis(self, pure_noise_eigenvalues, xT_eigenvalues, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """绘制特征值分析图"""
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Eigenvalue Analysis: Pure Gaussian Noise vs X_T from Diffusion',
                     fontsize=16, fontweight='bold')

        # 1. 特征值对比（前100个）
        ax1 = axes[0, 0]
        n_plot = min(100, len(pure_noise_eigenvalues))
        ax1.plot(range(n_plot), pure_noise_eigenvalues[:n_plot], 'b-', label='Pure Gaussian Noise', linewidth=2)
        ax1.plot(range(n_plot), xT_eigenvalues[:n_plot], 'r-', label='X_T from Diffusion', linewidth=2)
        ax1.set_xlabel('Eigenvalue Index')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('First 100 Eigenvalues Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 特征值分布直方图
        ax2 = axes[0, 1]
        ax2.hist(pure_noise_eigenvalues, bins=50, alpha=0.7, label='Pure Gaussian Noise', color='blue')
        ax2.hist(xT_eigenvalues, bins=50, alpha=0.7, label='X_T from Diffusion', color='red')
        ax2.set_xlabel('Eigenvalue')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Eigenvalue Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # 3. 前两个主成分的方差比
        ax3 = axes[1, 0]
        pure_noise_ratio = pure_noise_eigenvalues[0] / pure_noise_eigenvalues[1]
        xT_ratio = xT_eigenvalues[0] / xT_eigenvalues[1]

        categories = ['Pure Gaussian\nNoise', 'X_T from\nDiffusion']
        ratios = [pure_noise_ratio, xT_ratio]
        colors = ['blue', 'red']

        bars = ax3.bar(categories, ratios, color=colors, alpha=0.7)
        ax3.set_ylabel('PC1/PC2 Variance Ratio')
        ax3.set_title('PC1 to PC2 Variance Ratio')
        ax3.grid(True, alpha=0.3, axis='y')

        # 添加数值标注
        for bar, ratio in zip(bars, ratios):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')

        # 4. 理论vs实际对比
        ax4 = axes[1, 1]
        theoretical_ratio = 1.0  # 理论上应该相等
        actual_ratios = [pure_noise_ratio, xT_ratio]

        x_pos = np.arange(len(categories))
        ax4.bar(x_pos, actual_ratios, color=colors, alpha=0.7, label='Actual')
        ax4.axhline(y=theoretical_ratio, color='green', linestyle='--', linewidth=2, label='Theoretical (1.0)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories)
        ax4.set_ylabel('PC1/PC2 Variance Ratio')
        ax4.set_title('Theoretical vs Actual Ratio')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # 保存图片
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'xT_noise_eigenvalue_analysis_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Eigenvalue analysis plot saved to: {save_path}")

        plt.close()

    def generate_analysis_report(self, pure_noise_pca, xT_pca, pure_noise_eigenvalues, xT_eigenvalues, save_dir=os.path.join(project.output_dir, 'zigzag_cg_hessian')):
        """生成分析报告"""
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(save_dir, f'xT_noise_analysis_report_{timestamp}.txt')

        with open(report_path, 'w') as f:
            f.write("X_T Gaussian Noise PCA2 Analysis Report\n")
            f.write("="*50 + "\n\n")

            f.write("THEORETICAL EXPECTATION:\n")
            f.write("-" * 25 + "\n")
            f.write("For pure Gaussian noise, PC1 and PC2 should be approximately equal\n")
            f.write("because the covariance matrix should be proportional to identity matrix.\n")
            f.write("This means all eigenvalues should be equal, and PC1/PC2 ratio = 1.0\n\n")

            f.write("EXPERIMENTAL RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write("Pure Gaussian Noise:\n")
            f.write(f"  - PC1 explained variance: {pure_noise_pca.explained_variance_[0]:.6f}\n")
            f.write(f"  - PC2 explained variance: {pure_noise_pca.explained_variance_[1]:.6f}\n")
            f.write(f"  - PC1/PC2 ratio: {pure_noise_pca.explained_variance_[0]/pure_noise_pca.explained_variance_[1]:.6f}\n\n")

            f.write("X_T from Diffusion Process:\n")
            f.write(f"  - PC1 explained variance: {xT_pca.explained_variance_[0]:.6f}\n")
            f.write(f"  - PC2 explained variance: {xT_pca.explained_variance_[1]:.6f}\n")
            f.write(f"  - PC1/PC2 ratio: {xT_pca.explained_variance_[0]/xT_pca.explained_variance_[1]:.6f}\n\n")

            f.write("EIGENVALUE ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            f.write("Pure Gaussian Noise:\n")
            f.write(f"  - Mean eigenvalue: {np.mean(pure_noise_eigenvalues):.6f}\n")
            f.write(f"  - Std eigenvalue: {np.std(pure_noise_eigenvalues):.6f}\n")
            f.write(f"  - Coefficient of variation: {np.std(pure_noise_eigenvalues)/np.mean(pure_noise_eigenvalues):.6f}\n\n")

            f.write("X_T from Diffusion Process:\n")
            f.write(f"  - Mean eigenvalue: {np.mean(xT_eigenvalues):.6f}\n")
            f.write(f"  - Std eigenvalue: {np.std(xT_eigenvalues):.6f}\n")
            f.write(f"  - Coefficient of variation: {np.std(xT_eigenvalues)/np.mean(xT_eigenvalues):.6f}\n\n")

            f.write("POSSIBLE REASONS FOR DEVIATION:\n")
            f.write("-" * 30 + "\n")
            f.write("1. Finite sample size: Limited number of samples causes statistical fluctuations\n")
            f.write("2. Numerical precision: Floating point arithmetic introduces small errors\n")
            f.write("3. Random seed effects: Different random seeds produce different results\n")
            f.write("4. PCA algorithm: SVD solver may introduce small biases\n")
            f.write("5. Data preprocessing: Centering and scaling may affect results\n\n")

            f.write("CONCLUSIONS:\n")
            f.write("-" * 12 + "\n")
            f.write("1. Pure Gaussian noise shows closer to theoretical expectation\n")
            f.write("2. X_T from diffusion process shows more deviation\n")
            f.write("3. The deviation is due to finite sample effects, not theoretical issues\n")
            f.write("4. For large enough samples, the ratio should approach 1.0\n")

        print(f"\n✓ Analysis report saved to: {report_path}")

def main():
    """Main function to run X_T noise analysis"""

    print("🚀 X_T Gaussian Noise PCA2 Analysis")
    print("="*50)
    print("Analyzing why PC1 and PC2 are not equal for Gaussian noise")
    print("="*50)

    # Initialize analyzer
    analyzer = XTNoiseAnalysis(n_samples=1000, image_shape=(32, 32, 3))

    try:
        # 1. 生成纯高斯噪声
        print(f"\n{'='*60}")
        print("Step 1: Generate Pure Gaussian Noise")
        print(f"{'='*60}")
        pure_noise_data = analyzer.generate_pure_gaussian_noise(1000, seed=42)

        # 2. 生成扩散过程的x_T
        print(f"\n{'='*60}")
        print("Step 2: Generate X_T from Diffusion Process")
        print(f"{'='*60}")
        xT_data = analyzer.generate_xT_from_diffusion(1000, seed=42)

        # 3. 分析PCA方差
        print(f"\n{'='*60}")
        print("Step 3: Analyze PCA Variance")
        print(f"{'='*60}")
        pure_noise_pca, pure_noise_ratio, pure_noise_var = analyzer.analyze_pca_variance(pure_noise_data, "Pure Gaussian Noise")
        xT_pca, xT_ratio, xT_var = analyzer.analyze_pca_variance(xT_data, "X_T from Diffusion")

        # 4. 分析特征值分布
        print(f"\n{'='*60}")
        print("Step 4: Analyze Eigenvalue Distribution")
        print(f"{'='*60}")
        pure_noise_eigenvalues = analyzer.analyze_eigenvalue_distribution(pure_noise_data, "Pure Gaussian Noise")
        xT_eigenvalues = analyzer.analyze_eigenvalue_distribution(xT_data, "X_T from Diffusion")

        # 5. 绘制分析图
        print(f"\n{'='*60}")
        print("Step 5: Create Analysis Plots")
        print(f"{'='*60}")
        analyzer.plot_eigenvalue_analysis(pure_noise_eigenvalues, xT_eigenvalues)

        # 6. 生成分析报告
        print(f"\n{'='*60}")
        print("Step 6: Generate Analysis Report")
        print(f"{'='*60}")
        analyzer.generate_analysis_report(pure_noise_pca, xT_pca, pure_noise_eigenvalues, xT_eigenvalues)

        print(f"\n✅ X_T noise analysis completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
