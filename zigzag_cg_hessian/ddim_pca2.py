
# -*- coding: utf-8 -*-
"""
PCA → 投影 → 画 t=300/600/900 的 5 帧局部折线（DDIM-1000 轨迹）
要求：
- xt: numpy.ndarray, 形状 [1000, 32, 32, 3]
- 生成图：local_windows_ddim1000.png
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
from diffusers import DDPMPipeline, DDIMScheduler

import project as project

# ========== 1) 准备数据 ==========
# 方式 A：从文件读取（推荐）
# 把你的 DDIM-1000 轨迹保存成 .npy 后解除注释
# xt = np.load("xt_ddim1000.npy")  # shape: [1000, 32, 32, 3]

# 方式 B：如果还没有文件，这里放一个占位示例（请替换成自己的 xt）
# ------- ！！！实际使用时删除下面三行 -------
T = 1000
xt = np.random.randn(T, 32, 32, 3).astype(np.float32)  # 占位随机数据
# --------------------------------

assert isinstance(xt, np.ndarray), "xt 必须是 numpy 数组"
assert xt.ndim == 4 and xt.shape[0] >= 905 and xt.shape[1:] == (32, 32, 3), \
    f"xt 形状应为 [1000,32,32,3]，当前形状为 {xt.shape}"

# ========== 2) 拟合 PCA 并投影到 (PC1, PC2) ==========
# 将每帧展平为 3072 维，并用整条轨迹拟合 PCA（2 维）
X = xt.reshape(xt.shape[0], -1)              # [T, 3072]
pca = PCA(n_components=2, svd_solver='full') # 两个主轴
Z = pca.fit_transform(X)                     # [T, 2]；每一行就是 (z1, z2)

# ========== 3) 画局部 5 帧折线：t=300/600/900 ==========
def plot_local_window(Z2d: np.ndarray, t: int, ax: plt.Axes, title: str):
    """
    在给定 Axes 上绘制 [t-2, t-1, t, t+1, t+2] 的投影折线
    Z2d: [T, 2] 的投影坐标
    """
    assert 2 <= t <= Z2d.shape[0]-3, f"t={t} 超出可取 5 帧窗口的范围"
    idx = np.arange(t-2, t+3)
    pts = Z2d[idx]  # [5, 2]

    ax.plot(pts[:, 0], pts[:, 1], "-o", linewidth=2, markersize=5)
    # 用方块标注中心帧 t
    ax.scatter(pts[2, 0], pts[2, 1], s=120, marker='s', zorder=3, label=f"t={t}")
    ax.legend(frameon=False, loc="best")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.axis("equal")
    ax.grid(True, linestyle=":")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, t in zip(axes, [300, 600, 900]):
    plot_local_window(Z, t, ax, f"Local 5-step around t={t}")
plt.tight_layout()

out_path = "local_windows_ddim1000.png"
plt.savefig(out_path, dpi=200)
print(f"[OK] 已保存局部轨迹图：{os.path.abspath(out_path)}")
plt.show()
