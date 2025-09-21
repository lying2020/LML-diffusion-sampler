#!/usr/bin/env python3
"""
可视化 Hessian Free 和 CG 算法在 LML 中的使用
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_algorithm_flow_diagram():
    """创建算法流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 标题
    ax.text(5, 11.5, 'LML-Diffusion-Sampler 中的 Hessian Free 和 CG 算法', 
            fontsize=20, fontweight='bold', ha='center')
    
    # 1. 输入阶段
    input_box = FancyBboxPatch((0.5, 9.5), 2, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 10, '输入\n(noise_pred, x, t)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 2. LML 修正选择
    lml_box = FancyBboxPatch((3.5, 9.5), 3, 1, boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(lml_box)
    ax.text(5, 10, 'LML 修正方法选择', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 3. 三种方法
    # Original LML
    orig_box = FancyBboxPatch((0.5, 7.5), 2.5, 1, boxstyle="round,pad=0.1", 
                              facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(orig_box)
    ax.text(1.75, 8, 'Original LML\n(简化近似)', ha='center', va='center', fontsize=10)
    
    # Explicit Hessian
    expl_box = FancyBboxPatch((3.5, 7.5), 2.5, 1, boxstyle="round,pad=0.1", 
                              facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(expl_box)
    ax.text(4.75, 8, 'Explicit Hessian\n(显式计算)', ha='center', va='center', fontsize=10)
    
    # Hessian-Free
    hf_box = FancyBboxPatch((6.5, 7.5), 2.5, 1, boxstyle="round,pad=0.1", 
                            facecolor='lightpink', edgecolor='purple', linewidth=2)
    ax.add_patch(hf_box)
    ax.text(7.75, 8, 'Hessian-Free\n(CG + HVP)', ha='center', va='center', fontsize=10)
    
    # 4. Hessian-Free 详细流程
    # HVP 计算
    hvp_box = FancyBboxPatch((6, 5.5), 3, 1, boxstyle="round,pad=0.1", 
                             facecolor='lightcyan', edgecolor='cyan', linewidth=2)
    ax.add_patch(hvp_box)
    ax.text(7.5, 6, 'Hessian-Vector Product\n(Pearlmutter 方法)', ha='center', va='center', fontsize=10)
    
    # CG 算法
    cg_box = FancyBboxPatch((6, 4), 3, 1, boxstyle="round,pad=0.1", 
                            facecolor='lightsteelblue', edgecolor='navy', linewidth=2)
    ax.add_patch(cg_box)
    ax.text(7.5, 4.5, 'Conjugate Gradient\n(求解 Hx = b)', ha='center', va='center', fontsize=10)
    
    # 5. 输出
    output_box = FancyBboxPatch((3.5, 2), 3, 1, boxstyle="round,pad=0.1", 
                                facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 2.5, '修正后的噪声预测\n(corrected_noise)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 连接线
    # 输入到LML选择
    ax.arrow(2.5, 9.5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # LML选择到三种方法
    ax.arrow(4, 9.5, -1.25, -1, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(5, 9.5, 0, -1, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(6, 9.5, 1.25, -1, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Hessian-Free 内部流程
    ax.arrow(7.75, 7.5, 0, -1, head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2)
    ax.arrow(7.75, 5.5, 0, -1, head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2)
    
    # 所有方法到输出
    ax.arrow(1.75, 7.5, 1.75, -4.5, head_width=0.1, head_length=0.1, fc='orange', ec='orange')
    ax.arrow(4.75, 7.5, 0.25, -4.5, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax.arrow(7.75, 4, -2.25, -1.5, head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2)
    
    # 添加数学公式
    ax.text(1.5, 6.5, 'H⁻¹ ≈ (I + λI)⁻¹', ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='orange'))
    
    ax.text(4.75, 6.5, 'H = ∇²f(x)\n使用有限差分', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red'))
    
    ax.text(7.5, 3, 'Hv = ∇(∇f · v)\nCG: Hx = b', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='purple'))
    
    # 添加复杂度信息
    ax.text(0.5, 1, '复杂度对比:', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.5, 'Original: O(n) 内存, O(n) 计算\nExplicit: O(n²) 内存, O(n²) 计算\nHessian-Free: O(n) 内存, O(kn) 计算', 
            fontsize=10, va='top')
    
    plt.tight_layout()
    plt.savefig('hessian_cg_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_cg_algorithm_diagram():
    """创建CG算法详细流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 标题
    ax.text(5, 7.5, '共轭梯度算法 (Conjugate Gradient) 详细流程', 
            fontsize=16, fontweight='bold', ha='center')
    
    # 初始化
    init_box = FancyBboxPatch((1, 6), 2, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(init_box)
    ax.text(2, 6.4, '初始化\nx₀ = 0\nr₀ = b\np₀ = r₀', ha='center', va='center', fontsize=10)
    
    # 迭代循环
    loop_box = FancyBboxPatch((4, 5), 4, 2, boxstyle="round,pad=0.1", 
                              facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(loop_box)
    ax.text(6, 6.5, 'CG 迭代循环', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 循环内部步骤
    steps = [
        '1. 计算 Hp = H·p',
        '2. α = (r·r)/(p·Hp)',
        '3. x = x + α·p',
        '4. r = r - α·Hp',
        '5. 检查收敛 ||r|| < tol',
        '6. β = (r_new·r_new)/(r_old·r_old)',
        '7. p = r + β·p'
    ]
    
    for i, step in enumerate(steps):
        y_pos = 6.2 - i * 0.25
        ax.text(4.2, y_pos, step, fontsize=9, va='center')
    
    # 收敛检查
    conv_box = FancyBboxPatch((1, 2.5), 2, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(conv_box)
    ax.text(2, 2.9, '收敛?\n||r|| < tol', ha='center', va='center', fontsize=10)
    
    # 输出
    output_box = FancyBboxPatch((7, 2.5), 2, 0.8, boxstyle="round,pad=0.1", 
                                facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(output_box)
    ax.text(8, 2.9, '输出 x', ha='center', va='center', fontsize=10)
    
    # 连接线
    ax.arrow(3, 6, 1, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(6, 5, -3, -1.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(3, 2.9, 4, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # 添加数学公式
    ax.text(5, 1.5, '关键公式:', fontsize=12, fontweight='bold')
    ax.text(5, 1, 'α = rᵀr / pᵀHp\nβ = r_newᵀr_new / r_oldᵀr_old\nx = x + αp\nr = r - αHp', 
            fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig('cg_algorithm_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_pearlmutter_method_diagram():
    """创建Pearlmutter方法示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # 标题
    ax.text(7, 5.5, 'Pearlmutter 方法：Hessian-Vector Product 计算', 
            fontsize=16, fontweight='bold', ha='center')
    
    # 步骤1：计算梯度
    step1_box = FancyBboxPatch((0.5, 4), 3, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(step1_box)
    ax.text(2, 4.5, '步骤1: 计算梯度\ng = ∇f(x)', ha='center', va='center', fontsize=12)
    
    # 步骤2：计算内积
    step2_box = FancyBboxPatch((4.5, 4), 3, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(step2_box)
    ax.text(6, 4.5, '步骤2: 计算内积\ng·v', ha='center', va='center', fontsize=12)
    
    # 步骤3：计算二阶梯度
    step3_box = FancyBboxPatch((8.5, 4), 3, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(step3_box)
    ax.text(10, 4.5, '步骤3: 二阶梯度\nHv = ∇(g·v)', ha='center', va='center', fontsize=12)
    
    # 步骤4：输出
    step4_box = FancyBboxPatch((12.5, 4), 1, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(step4_box)
    ax.text(13, 4.5, 'Hv', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 连接线
    ax.arrow(3.5, 4.5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(7.5, 4.5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(11.5, 4.5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # 数学公式
    ax.text(2, 2.5, '代码实现:', fontsize=12, fontweight='bold')
    ax.text(2, 2, 'grad = torch.autograd.grad(log_prob, x_grad, create_graph=True)[0]', 
            fontsize=10, fontfamily='monospace')
    ax.text(2, 1.7, 'grad_dot_v = torch.sum(grad * v_grad)', 
            fontsize=10, fontfamily='monospace')
    ax.text(2, 1.4, 'hv = torch.autograd.grad(grad_dot_v, x_grad, retain_graph=True)[0]', 
            fontsize=10, fontfamily='monospace')
    
    # 优势说明
    ax.text(8, 2.5, '优势:', fontsize=12, fontweight='bold')
    ax.text(8, 2, '• 内存复杂度: O(n) 而不是 O(n²)', fontsize=10)
    ax.text(8, 1.7, '• 计算复杂度: O(n) 而不是 O(n²)', fontsize=10)
    ax.text(8, 1.4, '• 不需要存储完整的 Hessian 矩阵', fontsize=10)
    ax.text(8, 1.1, '• 适合大规模优化问题', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('pearlmutter_method_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    print("创建 Hessian Free 和 CG 算法可视化图表...")
    
    print("1. 创建算法流程图...")
    create_algorithm_flow_diagram()
    
    print("2. 创建CG算法详细流程图...")
    create_cg_algorithm_diagram()
    
    print("3. 创建Pearlmutter方法示意图...")
    create_pearlmutter_method_diagram()
    
    print("所有图表已保存完成！")
