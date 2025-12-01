import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as mpatches
import os
import project
from project import results_iclr_path

from plot_utils import (
    data_colors, data_labels, data_markers
)

import numpy as np

def create_custom_legend():

    # 创建自定义图例的标记，包含颜色和marker
    # 将线条和marker分开绘制：先画线条，再在末尾单独画marker
    custom_lines = []
    for color, marker in zip(data_colors, data_markers):
        # 创建线条（不带marker）
        line = Line2D([0, 4], [0, 0], color=color, lw=4, linestyle='-')
        # 创建marker点（在末尾位置，不画线）
        marker_point = Line2D([4], [0], color=color, lw=0, marker=marker,
                              markersize=10, linestyle='None')
        # 将线条和marker组合成一个元组
        custom_lines.append((line, marker_point))

    # 创建只有图例的图形
    fig = plt.figure(figsize=(10, 0.5))  # 增加宽度以容纳更宽的间距
    # 使用HandlerTuple来处理组合的线条和marker
    handler_map = {tuple: HandlerTuple(ndivide=None)}
    fig.legend(custom_lines, data_labels, loc='center', ncol=len(data_labels),
               frameon=False, columnspacing=1.5, handletextpad=0.8,
               handler_map=handler_map)

    # 移除坐标轴
    plt.axis('off')

    return fig

# 创建并保存图例
fig = create_custom_legend()
plt.savefig(os.path.join(results_iclr_path, 'fid_legend.jpg'), bbox_inches='tight', dpi=300)
plt.close(fig)