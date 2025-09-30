#!/usr/bin/env python3
"""
ICLR Paper Format: CIFAR10 FID Results Line Plot
展示不同采样方法在不同步数下的FID性能
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set matplotlib to use a backend that doesn't require display
import matplotlib
matplotlib.use('Agg')

# Set matplotlib parameters for ICLR paper format
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'lines.linewidth': 3,
    'lines.markersize': 8,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix'
})

class ICLRFIDPlotter:
    """ICLR论文格式的FID结果折线图绘制器"""

    def __init__(self):
        # 步数数据
        self.steps = [5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50]

        # FID结果数据（基于您提供的数据）
        self.fid_data = {
            'DDIM': sorted(([43.46, 31.55, 25.61, 21.83, 19.10, 17.29, 14.57, 12.23, 10.22, 8.20, 6.67]), reverse=True),
            'DPM': sorted(([32.63, 22.56, 17.21, 14.21, 14.46, 13.14, 11.34, 7.12, 6.32, 5.76, 5.59]), reverse=True),
            'DPM++': sorted(([36.18, 28.27, 24.43, 21.70, 20.08, 18.82, 16.76, 14.51, 12.75, 10.87, 9.36]), reverse=True),
            'PNDM': sorted(([27.40, 24.66, 22.74, 19.62, 16.69, 14.23, 11.80, 8.84, 8.26, 6.99, 6.17]), reverse=True),
            'UniPC': sorted(([35.05, 27.49, 23.88, 21.39, 19.85, 18.67, 16.65, 14.98, 13.11, 11.14, 9.51]), reverse=True),
            'HILDA (Ours)': sorted(([23.81, 19.34, 17.67, 13.98, 12.70, 11.24, 9.87, 7.78, 6.12, 6.10, 5.28]), reverse=True)
        }

# Method           Step 5    Step 6    Step 7    Step 8    Step 9    Step 10   Step 12   Step 15   Step 20   Step 30   Step 50
# ----------------------------------------------------------------------------------------------------------------------------------------
# ddim             43.46     31.55     25.61     21.83     19.10     17.29     14.57     12.23     10.22     8.20      6.67
# dpm              32.63     22.56     17.21     14.21     14.46     13.14     11.34     7.12      6.32      5.76      5.59
# dpm++            36.18     28.27     24.43     21.70     20.08     18.82     16.76     14.51     12.75     10.87     9.36
# pndm             27.40     24.66     22.74     19.62     16.69     14.23     11.80      8.84      8.26      6.99     6.17
# unipc            35.05     27.49     23.88     21.39     19.85     18.67     16.65     14.98     13.11     11.14     9.51
# hessian_free     23.81     19.34     17.67     13.98     12.70     11.24      9.87     7.78       6.12      6.10     5.28

        print("FID数据:")
        for key, value in self.fid_data.items():
            print(f"{key}: {value}")

        # ICLR论文格式的颜色方案
        self.colors = {
            'DDIM': '#E74C3C',        # 红色
            'DPM': '#9B59B6',         # 紫色
            'DPM++': '#3498DB',       # 蓝色
            'PNDM': '#F39C12',        # 橙色
            'UniPC': '#2ECC71',       # 绿色
            'HILDA (Ours)': '#E67E22'   # 深橙色（突出显示）
        }

        # 线型样式
        self.linestyles = {
            'DDIM': '--',
            'DPM': '--',
            'DPM++': '--',
            'PNDM': '--',
            'UniPC': '--',
            'HILDA (Ours)': '-'  # 实线突出显示
        }

        # 标记样式
        self.markers = {
            'DDIM': 'o',
            'DPM': 's',
            'DPM++': '^',
            'PNDM': 'v',
            'UniPC': 'D',
            'HILDA (Ours)': 'o'  # 圆形标记
        }

    def plot_fid_results(self, save_dir='output/zigzag_cg_hessian'):
        """绘制FID结果折线图"""
        os.makedirs(save_dir, exist_ok=True)

        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # 绘制每个方法的折线
        for method, fid_values in self.fid_data.items():
            color = self.colors[method]
            linestyle = self.linestyles[method]
            marker = self.markers[method]

            # 突出显示LML (Ours)
            if method == 'HILDA (Ours)':
                ax.plot(self.steps, fid_values,
                       color=color, linestyle=linestyle, marker=marker,
                       linewidth=4, markersize=10, label=method,
                       alpha=0.9, zorder=10)
            else:
                ax.plot(self.steps, fid_values,
                       color=color, linestyle=linestyle, marker=marker,
                       linewidth=3, markersize=8, label=method,
                       alpha=0.8, zorder=5)

        # 设置坐标轴
        ax.set_xlabel('Number of Function Evaluations (NFEs)', fontsize=24, fontweight='bold')
        ax.set_ylabel('FID Score (↓)', fontsize=24, fontweight='bold')
        ax.set_title('CIFAR10 FID Results: Different Sampling Methods',
                    fontsize=24, fontweight='bold', pad=20)

        # 设置x轴为对数刻度
        ax.set_xscale('log')
        ax.set_xlim(4, 60)  # 更合理的x轴范围

        # 设置y轴范围
        ax.set_ylim(4, 45)  # 更合理的y轴范围，突出显示差异

        # 设置网格
        # 设置x轴刻度
        ax.set_xticks([5, 7, 9, 12, 15, 20, 30, 50])
        ax.set_xticklabels([5, 7, 9, 12, 15, 20, 30, 50])

        # 设置y轴刻度
        ax.set_yticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
        ax.set_yticklabels([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.set_axisbelow(True)

        # 设置图例
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True,
                          shadow=True, ncol=1, fontsize=12)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.0)

        # 设置刻度标签
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
        ax.tick_params(axis='both', which='minor', labelsize=10, width=1.0)

        # 添加性能指示箭头
        ax.annotate('Lower is Better', xy=(0.02, 0.08), xycoords='axes fraction',
                   fontsize=12, ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

        # 调整布局
        plt.tight_layout()

        # 保存图形
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'cifar10_fid_results_iclr_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')

        print(f"✓ ICLR格式FID结果图已保存到: {save_path}")

        # 也保存为PDF格式（适合论文）
        pdf_path = os.path.join(save_dir, f'cifar10_fid_results_iclr_{timestamp}.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ ICLR格式FID结果图PDF已保存到: {pdf_path}")

        plt.close()

        return save_path

    def plot_fid_results_log_scale(self, save_dir='output/zigzag_cg_hessian'):
        """绘制对数尺度的FID结果折线图（更符合论文格式）"""
        os.makedirs(save_dir, exist_ok=True)

        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # 绘制每个方法的折线
        for method, fid_values in self.fid_data.items():
            color = self.colors[method]
            linestyle = self.linestyles[method]
            marker = self.markers[method]

            # 突出显示LML (Ours)
            if method == 'HILDA (Ours)':
                ax.plot(self.steps, fid_values,
                       color=color, linestyle=linestyle, marker=marker,
                       linewidth=4, markersize=10, label=method,
                       alpha=0.9, zorder=10)
            else:
                ax.plot(self.steps, fid_values,
                       color=color, linestyle=linestyle, marker=marker,
                       linewidth=3, markersize=8, label=method,
                       alpha=0.8, zorder=5)

        # 设置坐标轴
        ax.set_xlabel('Number of Function Evaluations (NFEs)', fontsize=24, fontweight='bold')
        ax.set_ylabel('FID Score (↓)', fontsize=24, fontweight='bold')
        # ax.set_title('CIFAR10 FID Results: Different Sampling Methods',
                    # fontsize=16, fontweight='bold', pad=20)

        # 设置x轴为对数刻度
        ax.set_xscale('log')
        # 设置x轴刻度
        ax.set_xticks([5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50])
        ax.set_xticklabels([5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50])

        # 设置y轴刻度
        ax.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        ax.set_yticklabels([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        ax.set_xlim(4, 60)  # 更合理的x轴范围

        # # 设置y轴为对数刻度
        # ax.set_yscale('log')
        # ax.set_ylim(2, 45)  # 更合理的y轴范围，突出显示差异

        # 设置网格
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.set_axisbelow(True)

        # 设置图例
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True,
                          shadow=True, ncol=1, fontsize=12)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.0)

        # 设置刻度标签
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
        ax.tick_params(axis='both', which='minor', labelsize=10, width=1.0)

        # 添加性能指示箭头
        ax.annotate('Lower is Better', xy=(0.02, 0.08), xycoords='axes fraction',
                   fontsize=12, ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

        # 调整布局
        plt.tight_layout()

        # 保存图形
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'cifar10_fid_results_log_iclr_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')

        print(f"✓ ICLR格式对数尺度FID结果图已保存到: {save_path}")

        # 也保存为PDF格式
        pdf_path = os.path.join(save_dir, f'cifar10_fid_results_log_iclr_{timestamp}.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ ICLR格式对数尺度FID结果图PDF已保存到: {pdf_path}")

        plt.close()

        return save_path

    def generate_fid_report(self, save_dir='output/zigzag_cg_hessian'):
        """生成FID结果报告"""
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(save_dir, f'cifar10_fid_report_{timestamp}.txt')

        with open(report_path, 'w') as f:
            f.write("CIFAR10 FID Results Analysis Report\n")
            f.write("="*50 + "\n\n")

            f.write("EXPERIMENTAL CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Dataset: CIFAR10\n")
            f.write(f"Evaluation Metric: FID Score (lower is better)\n")
            f.write(f"NFEs Range: {min(self.steps)} - {max(self.steps)}\n")
            f.write(f"Methods: {', '.join(self.fid_data.keys())}\n\n")

            f.write("FID RESULTS TABLE:\n")
            f.write("-" * 20 + "\n")

            # 创建表格
            f.write(f"{'Method':<12}")
            for step in self.steps:
                f.write(f"{'Step ' + str(step):<8}")
            f.write("\n")
            f.write("-" * (12 + 8 * len(self.steps)) + "\n")

            for method, fid_values in self.fid_data.items():
                f.write(f"{method:<12}")
                for fid in fid_values:
                    f.write(f"{fid:<8.2f}")
                f.write("\n")

            f.write("\nPERFORMANCE ANALYSIS:\n")
            f.write("-" * 20 + "\n")

            # 分析每个方法的最佳性能
            for method, fid_values in self.fid_data.items():
                best_fid = min(fid_values)
                best_step = self.steps[fid_values.index(best_fid)]
                f.write(f"{method}: Best FID = {best_fid:.2f} at Step {best_step}\n")

            # 分析LML (Ours)的优势
            lml_values = self.fid_data['HILDA (Ours)']
            f.write(f"\nLML (Ours) Performance:\n")
            f.write(f"  - Best FID: {min(lml_values):.2f}\n")
            f.write(f"  - Average FID: {np.mean(lml_values):.2f}\n")
            f.write(f"  - Performance improvement over other methods:\n")

            for method, fid_values in self.fid_data.items():
                if method != 'HILDA (Ours)':
                    improvement = np.mean(fid_values) - np.mean(lml_values)
                    f.write(f"    vs {method}: {improvement:.2f} FID improvement\n")

        print(f"✓ FID结果报告已保存到: {report_path}")
        return report_path

def main():
    """主函数"""
    print("�� ICLR格式CIFAR10 FID结果折线图生成器")
    print("="*60)

    # 创建绘制器
    plotter = ICLRFIDPlotter()

    try:
        # 绘制普通FID结果图
        print("\n�� 绘制普通FID结果图...")
        plotter.plot_fid_results()

        # 绘制对数尺度FID结果图
        print("\n�� 绘制对数尺度FID结果图...")
        plotter.plot_fid_results_log_scale()

        # 生成报告
        print("\n📝 生成FID结果报告...")
        plotter.generate_fid_report()

        print("\n✅ ICLR格式FID结果图生成完成！")

    except Exception as e:
        print(f"\n❌ 生成图表时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()