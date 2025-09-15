#!/usr/bin/env python3
"""
Create Metrics Explanation Chart

This script creates a visual explanation of how the metrics are defined and calculated.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

def create_metrics_explanation_chart():
    """Create a comprehensive chart explaining all metrics"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('算法对比指标定义详解', fontsize=24, fontweight='bold', y=0.95)
    
    # 1. 质量指标 (Quality)
    ax1 = plt.subplot(3, 3, 1)
    ax1.set_title('🎨 质量指标 (Quality)', fontsize=16, fontweight='bold', pad=20)
    ax1.axis('off')
    
    quality_text = """
    主要指标: 图像方差 (variance)
    
    计算公式:
    variance = mean([var(img) for img in images])
    
    子指标:
    • 清晰度: 拉普拉斯方差
    • 对比度: RMS对比度  
    • 颜色多样性: RGB通道差异
    
    范围: 0 → ∞
    越高越好: 表示图像内容越丰富
    """
    
    ax1.text(0.05, 0.9, quality_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # 2. 效率指标 (Efficiency)
    ax2 = plt.subplot(3, 3, 2)
    ax2.set_title('⚖️ 效率指标 (Efficiency)', fontsize=16, fontweight='bold', pad=20)
    ax2.axis('off')
    
    efficiency_text = """
    计算公式:
    efficiency = avg_variance / avg_time_per_image
    
    物理意义:
    每秒能生成多少质量分数
    
    单位: quality/s
    
    示例:
    Hessian-Free-Fast: 301089.83 quality/s
    表示每秒生成301089.83个质量分数
    
    越高越好: 表示算法效率越高
    """
    
    ax2.text(0.05, 0.9, efficiency_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    # 3. 稳定性指标 (Stability)
    ax3 = plt.subplot(3, 3, 3)
    ax3.set_title('🔒 稳定性指标 (Stability)', fontsize=16, fontweight='bold', pad=20)
    ax3.axis('off')
    
    stability_text = """
    时间稳定性:
    time_stability = 1 / (1 + std_time / avg_time)
    
    质量稳定性:
    quality_stability = 1 / (1 + std_quality / avg_quality)
    
    范围: 0 → 1
    1 = 完全稳定
    0 = 完全不稳定
    
    示例:
    time_stability = 0.984
    表示时间变化很小，很稳定
    
    越高越好: 表示性能越一致
    """
    
    ax3.text(0.05, 0.9, stability_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
    
    # 4. 速度指标 (Speed)
    ax4 = plt.subplot(3, 3, 4)
    ax4.set_title('⚡ 速度指标 (Speed)', fontsize=16, fontweight='bold', pad=20)
    ax4.axis('off')
    
    speed_text = """
    主要指标: 平均生成时间
    
    计算方法:
    • 记录每个图像的生成时间
    • 计算统计指标
    
    统计指标:
    • avg_time_per_image: 平均时间
    • std_time_per_image: 时间标准差
    • min_time_per_image: 最短时间
    • max_time_per_image: 最长时间
    
    单位: 秒 (s)
    越低越好: 表示算法越快
    """
    
    ax4.text(0.05, 0.9, speed_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.8))
    
    # 5. FID指标
    ax5 = plt.subplot(3, 3, 5)
    ax5.set_title('📊 FID指标 (Fréchet Inception Distance)', fontsize=16, fontweight='bold', pad=20)
    ax5.axis('off')
    
    fid_text = """
    定义: 生成图像与真实图像的分布距离
    
    计算公式:
    FID = ||μ₁ - μ₂||² + Tr(Σ₁) + Tr(Σ₂) - 2Tr(√(Σ₁Σ₂))
    
    其中:
    • μ₁, μ₂: 特征均值
    • Σ₁, Σ₂: 特征协方差矩阵
    
    范围: 0 → ∞
    0 = 完美匹配
    < 50 = 很好
    50-100 = 好
    > 100 = 需要改进
    
    越低越好: 表示与真实图像越相似
    """
    
    ax5.text(0.05, 0.9, fid_text, transform=ax5.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='plum', alpha=0.8))
    
    # 6. 指标关系图
    ax6 = plt.subplot(3, 3, 6)
    ax6.set_title('📈 指标关系图', fontsize=16, fontweight='bold', pad=20)
    ax6.axis('off')
    
    # 创建指标关系图
    metrics = ['质量\n(Quality)', '速度\n(Speed)', '效率\n(Efficiency)', '稳定性\n(Stability)', 'FID\n(FID)']
    positions = [(0.2, 0.8), (0.8, 0.8), (0.5, 0.5), (0.2, 0.2), (0.8, 0.2)]
    colors = ['lightblue', 'gold', 'lightgreen', 'lightcoral', 'plum']
    
    for i, (metric, pos, color) in enumerate(zip(metrics, positions, colors)):
        # 绘制指标框
        box = FancyBboxPatch((pos[0]-0.1, pos[1]-0.1), 0.2, 0.2,
                            boxstyle="round,pad=0.02", facecolor=color, alpha=0.8)
        ax6.add_patch(box)
        ax6.text(pos[0], pos[1], metric, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 绘制关系箭头
    # 效率 = 质量 / 速度
    ax6.annotate('', xy=(0.5, 0.5), xytext=(0.2, 0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax6.annotate('', xy=(0.5, 0.5), xytext=(0.8, 0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax6.text(0.5, 0.6, '效率 = 质量 / 速度', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='red')
    
    # 7. 计算流程图
    ax7 = plt.subplot(3, 3, 7)
    ax7.set_title('🔄 计算流程图', fontsize=16, fontweight='bold', pad=20)
    ax7.axis('off')
    
    flow_text = """
    1. 图像生成
       ↓
    2. 质量计算
       • 方差计算
       • 清晰度计算
       • 对比度计算
       ↓
    3. 时间测量
       • 记录生成时间
       • 计算统计指标
       ↓
    4. 效率计算
       • efficiency = quality / time
       ↓
    5. 稳定性计算
       • time_stability = 1/(1+std/mean)
       • quality_stability = 1/(1+std/mean)
       ↓
    6. FID计算
       • 特征提取
       • 分布距离计算
    """
    
    ax7.text(0.05, 0.9, flow_text, transform=ax7.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # 8. 应用场景选择
    ax8 = plt.subplot(3, 3, 8)
    ax8.set_title('🎯 应用场景选择', fontsize=16, fontweight='bold', pad=20)
    ax8.axis('off')
    
    scenario_text = """
    实时应用:
    • 主要关注: 速度、效率
    • 推荐: Hessian-Free-Fast
    
    高质量生成:
    • 主要关注: 质量、FID
    • 推荐: Hessian-Free-Fast
    
    生产环境:
    • 主要关注: 稳定性、效率
    • 推荐: LML-Improved
    
    研究应用:
    • 主要关注: 质量、FID
    • 推荐: Hessian-Free-Advanced
    """
    
    ax8.text(0.05, 0.9, scenario_text, transform=ax8.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # 9. 指标权重
    ax9 = plt.subplot(3, 3, 9)
    ax9.set_title('⚖️ 指标权重', fontsize=16, fontweight='bold', pad=20)
    ax9.axis('off')
    
    # 创建权重饼图
    weights = [30, 25, 20, 15, 10]  # 效率、质量、速度、稳定性、FID
    labels = ['效率\n30%', '质量\n25%', '速度\n20%', '稳定性\n15%', 'FID\n10%']
    colors = ['lightgreen', 'lightblue', 'gold', 'lightcoral', 'plum']
    
    wedges, texts, autotexts = ax9.pie(weights, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 10})
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # 保存图表
    plt.savefig('metrics_definition_explanation.png', dpi=300, bbox_inches='tight')
    print("指标定义说明图表已保存: metrics_definition_explanation.png")

if __name__ == '__main__':
    create_metrics_explanation_chart()
