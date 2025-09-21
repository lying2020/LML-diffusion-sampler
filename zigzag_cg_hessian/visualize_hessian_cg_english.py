#!/usr/bin/env python3
"""
Visualize Hessian Free and CG algorithms in LML
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

import project as project


zigzag_cg_hessian_dir = os.path.join(project.output_dir, 'zigzag_cg_hessian')
if not os.path.exists(zigzag_cg_hessian_dir):
    os.makedirs(zigzag_cg_hessian_dir)

def create_algorithm_flow_diagram():
    """Create algorithm flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(5, 11.5, 'Hessian Free and CG Algorithms in LML-Diffusion-Sampler',
            fontsize=20, fontweight='bold', ha='center')

    # 1. Input stage
    input_box = FancyBboxPatch((0.5, 9.5), 2, 1, boxstyle="round,pad=0.1",
                               facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 10, 'Input\n(noise_pred, x, t)', ha='center', va='center', fontsize=12, fontweight='bold')

    # 2. LML correction selection
    lml_box = FancyBboxPatch((3.5, 9.5), 3, 1, boxstyle="round,pad=0.1",
                             facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(lml_box)
    ax.text(5, 10, 'LML Correction Method Selection', ha='center', va='center', fontsize=12, fontweight='bold')

    # 3. Three methods
    # Original LML
    orig_box = FancyBboxPatch((0.5, 7.5), 2.5, 1, boxstyle="round,pad=0.1",
                              facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(orig_box)
    ax.text(1.75, 8, 'Original LML\n(Simplified Approximation)', ha='center', va='center', fontsize=10)

    # Explicit Hessian
    expl_box = FancyBboxPatch((3.5, 7.5), 2.5, 1, boxstyle="round,pad=0.1",
                              facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(expl_box)
    ax.text(4.75, 8, 'Explicit Hessian\n(Direct Computation)', ha='center', va='center', fontsize=10)

    # Hessian-Free
    hf_box = FancyBboxPatch((6.5, 7.5), 2.5, 1, boxstyle="round,pad=0.1",
                            facecolor='lightpink', edgecolor='purple', linewidth=2)
    ax.add_patch(hf_box)
    ax.text(7.75, 8, 'Hessian-Free\n(CG + HVP)', ha='center', va='center', fontsize=10)

    # 4. Hessian-Free detailed flow
    # HVP calculation
    hvp_box = FancyBboxPatch((6, 5.5), 3, 1, boxstyle="round,pad=0.1",
                             facecolor='lightcyan', edgecolor='cyan', linewidth=2)
    ax.add_patch(hvp_box)
    ax.text(7.5, 6, 'Hessian-Vector Product\n(Pearlmutter Method)', ha='center', va='center', fontsize=10)

    # CG algorithm
    cg_box = FancyBboxPatch((6, 4), 3, 1, boxstyle="round,pad=0.1",
                            facecolor='lightsteelblue', edgecolor='navy', linewidth=2)
    ax.add_patch(cg_box)
    ax.text(7.5, 4.5, 'Conjugate Gradient\n(Solve Hx = b)', ha='center', va='center', fontsize=10)

    # 5. Output
    output_box = FancyBboxPatch((3.5, 2), 3, 1, boxstyle="round,pad=0.1",
                                facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 2.5, 'Corrected Noise Prediction\n(corrected_noise)', ha='center', va='center', fontsize=12, fontweight='bold')

    # Connection lines
    # Input to LML selection
    ax.arrow(2.5, 9.5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

    # LML selection to three methods
    ax.arrow(4, 9.5, -1.25, -1, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(5, 9.5, 0, -1, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(6, 9.5, 1.25, -1, head_width=0.1, head_length=0.1, fc='black', ec='black')

    # Hessian-Free internal flow
    ax.arrow(7.75, 7.5, 0, -1, head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2)
    ax.arrow(7.75, 5.5, 0, -1, head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2)

    # All methods to output
    ax.arrow(1.75, 7.5, 1.75, -4.5, head_width=0.1, head_length=0.1, fc='orange', ec='orange')
    ax.arrow(4.75, 7.5, 0.25, -4.5, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax.arrow(7.75, 4, -2.25, -1.5, head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2)

    # Add mathematical formulas
    ax.text(1.5, 6.5, 'H⁻¹ ≈ (I + λI)⁻¹', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='orange'))

    ax.text(4.75, 6.5, 'H = ∇²f(x)\nUsing Finite Differences', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red'))

    ax.text(7.5, 3, 'Hv = ∇(∇f · v)\nCG: Hx = b', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='purple'))

    # Add complexity information
    ax.text(0.5, 1, 'Complexity Comparison:', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.5, 'Original: O(n) memory, O(n) computation\nExplicit: O(n²) memory, O(n²) computation\nHessian-Free: O(n) memory, O(kn) computation',
            fontsize=10, va='top')

    plt.tight_layout()
    plt.savefig(os.path.join(zigzag_cg_hessian_dir, 'hessian_cg_flow_diagram_english.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_cg_algorithm_diagram():
    """Create CG algorithm detailed flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 7.5, 'Conjugate Gradient Algorithm Detailed Flow',
            fontsize=16, fontweight='bold', ha='center')

    # Initialization
    init_box = FancyBboxPatch((1, 6), 2, 0.8, boxstyle="round,pad=0.1",
                              facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(init_box)
    ax.text(2, 6.4, 'Initialize\nx₀ = 0\nr₀ = b\np₀ = r₀', ha='center', va='center', fontsize=10)

    # Iteration loop
    loop_box = FancyBboxPatch((4, 5), 4, 2, boxstyle="round,pad=0.1",
                              facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(loop_box)
    ax.text(6, 6.5, 'CG Iteration Loop', ha='center', va='center', fontsize=12, fontweight='bold')

    # Loop internal steps
    steps = [
        '1. Compute Hp = H·p',
        '2. α = (r·r)/(p·Hp)',
        '3. x = x + α·p',
        '4. r = r - α·Hp',
        '5. Check convergence ||r|| < tol',
        '6. β = (r_new·r_new)/(r_old·r_old)',
        '7. p = r + β·p'
    ]

    for i, step in enumerate(steps):
        y_pos = 6.2 - i * 0.25
        ax.text(4.2, y_pos, step, fontsize=9, va='center')

    # Convergence check
    conv_box = FancyBboxPatch((1, 2.5), 2, 0.8, boxstyle="round,pad=0.1",
                              facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(conv_box)
    ax.text(2, 2.9, 'Converged?\n||r|| < tol', ha='center', va='center', fontsize=10)

    # Output
    output_box = FancyBboxPatch((7, 2.5), 2, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(output_box)
    ax.text(8, 2.9, 'Output x', ha='center', va='center', fontsize=10)

    # Connection lines
    ax.arrow(3, 6, 1, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(6, 5, -3, -1.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(3, 2.9, 4, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

    # Add mathematical formulas
    ax.text(5, 1.5, 'Key Formulas:', fontsize=12, fontweight='bold')
    ax.text(5, 1, 'α = rᵀr / pᵀHp\nβ = r_newᵀr_new / r_oldᵀr_old\nx = x + αp\nr = r - αHp',
            fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'))

    plt.tight_layout()
    plt.savefig(os.path.join(zigzag_cg_hessian_dir, 'cg_algorithm_diagram_english.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_pearlmutter_method_diagram():
    """Create Pearlmutter method diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(7, 5.5, 'Pearlmutter Method: Hessian-Vector Product Computation',
            fontsize=16, fontweight='bold', ha='center')

    # Step 1: Compute gradient
    step1_box = FancyBboxPatch((0.5, 4), 3, 1, boxstyle="round,pad=0.1",
                               facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(step1_box)
    ax.text(2, 4.5, 'Step 1: Compute Gradient\ng = ∇f(x)', ha='center', va='center', fontsize=12)

    # Step 2: Compute dot product
    step2_box = FancyBboxPatch((4.5, 4), 3, 1, boxstyle="round,pad=0.1",
                               facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(step2_box)
    ax.text(6, 4.5, 'Step 2: Compute Dot Product\ng·v', ha='center', va='center', fontsize=12)

    # Step 3: Compute second-order gradient
    step3_box = FancyBboxPatch((8.5, 4), 3, 1, boxstyle="round,pad=0.1",
                               facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(step3_box)
    ax.text(10, 4.5, 'Step 3: Second-order Gradient\nHv = ∇(g·v)', ha='center', va='center', fontsize=12)

    # Step 4: Output
    step4_box = FancyBboxPatch((12.5, 4), 1, 1, boxstyle="round,pad=0.1",
                               facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(step4_box)
    ax.text(13, 4.5, 'Hv', ha='center', va='center', fontsize=12, fontweight='bold')

    # Connection lines
    ax.arrow(3.5, 4.5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(7.5, 4.5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(11.5, 4.5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

    # Mathematical formulas
    ax.text(2, 2.5, 'Code Implementation:', fontsize=12, fontweight='bold')
    ax.text(2, 2, 'grad = torch.autograd.grad(log_prob, x_grad, create_graph=True)[0]',
            fontsize=10, fontfamily='monospace')
    ax.text(2, 1.7, 'grad_dot_v = torch.sum(grad * v_grad)',
            fontsize=10, fontfamily='monospace')
    ax.text(2, 1.4, 'hv = torch.autograd.grad(grad_dot_v, x_grad, retain_graph=True)[0]',
            fontsize=10, fontfamily='monospace')

    # Advantages
    ax.text(8, 2.5, 'Advantages:', fontsize=12, fontweight='bold')
    ax.text(8, 2, '• Memory complexity: O(n) instead of O(n²)', fontsize=10)
    ax.text(8, 1.7, '• Computational complexity: O(n) instead of O(n²)', fontsize=10)
    ax.text(8, 1.4, '• No need to store complete Hessian matrix', fontsize=10)
    ax.text(8, 1.1, '• Suitable for large-scale optimization problems', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(zigzag_cg_hessian_dir, 'pearlmutter_method_diagram_english.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    print("Creating Hessian Free and CG algorithm visualization charts...")

    print("1. Creating algorithm flow diagram...")
    create_algorithm_flow_diagram()

    print("2. Creating CG algorithm detailed flow diagram...")
    create_cg_algorithm_diagram()

    print("3. Creating Pearlmutter method diagram...")
    create_pearlmutter_method_diagram()

    print("All charts saved successfully!")
