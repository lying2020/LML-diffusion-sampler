"""
Shared plotting utilities for all plot files.
This module contains common functions and classes used across different plotting scripts.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免程序卡住
import matplotlib.pyplot as plt
import matplotlib.transforms
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import os
import sys

# Add project path
import project
from project import results_iclr_path, current_path

# 定义颜色和标签
# data_colors = ['magenta', 'red', 'blue', 'orange', 'olive', 'green']
# 定义颜色和标签 - 使用两组渐变色和橘黄色
# DDIM, PNDM, DPM-Solver 使用蓝色渐变: ['#D1E9FC', '#85C1E9', '#5DADE2']
# SMM, UniPC 使用绿色渐变: ['#D5F4E6', '#82E0AA']
# AE-VP 使用橘黄色: '#FFA500'
# data_colors = ['blue', 'red', 'orange', 'purple', 'green', 'olive', 'brown', 'magenta', 'cyan', 'crimson', 'gray', 'black']
# ICLR paper color scheme: monochromatic blue tones with teal and dark gray
# Index mapping: [0: DDIM, 1: PNDM, 2: DPM-Solver, 3: DPM-Solver++, 4: UniPC, 5: GeoDiff(Ours)]
# Color scheme from provided image:
# 0: 珊瑚粉 (Coral Pink) #FF6B6B
# 1: 钴蓝色 (Cobalt Blue) #4A6FA7
# 2: 柠檬黄 (Lemon Yellow) #FFD166
# 3: 电光紫 (Electric Purple) #9B5DE5
# 4: 清新绿 (Fresh Green) #06D6A0
# 5: 深橙色 (Deep Orange) #FF8C42 - 新增颜色，与现有配色协调
# data_colors = ['#FF6B6B', '#4A6FA7', '#FFD166', '#9B5DE5', '#06D6A0', '#FF8C42']
# Color scheme from provided image (blue gradient from dark to light):
# 0: DDIM - #1A237E (深蓝色)
# 1: PNDM - #283593 (中深蓝色)
# 2: DPM-Solver - #303F9F (中深蓝色)
# 3: DPM-Solver++ - #5C6BC0 (中蓝色)
# 4: UniPC - #9FA8DA (浅蓝/淡紫色)
# 5: GeoDiff(Ours) - #EAEAF4 (极浅，接近白色)
# data_colors = ['#1A237E', '#303F9F', '#5C6BC0', '#9FA8DA', '#EAEAF4', '#283593']

# Color scheme from provided gradient images:
# 调整颜色顺序，让 GeoDiff (Ours) 使用最醒目的颜色
# 0: DDIM - 青绿色 (Teal Green) #2ad4af
# 1: PNDM - 天蓝色 (Sky Blue) #27a6cc
# 2: DPM-Solver - 青柠绿 (Lime Green) #98be2c
# 3: DPM-Solver++ - 深绿色 (Deep Green) #00b168
# 4: UniPC - 粉红色 (Pink) #fcc5c5
# 5: GeoDiff (Ours) - 橙色 (Orange) #fcbd60 - 最醒目的颜色
data_colors = ['#2ad4af', '#27a6cc', '#98be2c', '#00b168', '#fcc5c5', '#fcbd60']

data_labels = ['DDIM', 'PNDM', 'DPM-Solver', 'DPM-Solver++', 'UniPC', 'GeoDiff(Ours)']
# data_markers = ['s', 'o', 'D', 'v', '*', 'p']
# Marker options: 'o' (circle), 's' (square), 'D' (diamond), 'v' (triangle down),
# '^' (triangle up), 'h' (hexagon), 'H' (hexagon2), '8' (octagon), 'p' (pentagon),
# 'P' (plus filled), 'X' (x filled), '+' (plus), 'x' (x)
data_markers = ['o', 's', 'D', 'v', '^', 'p']  # 将 '*' 替换为 '^' (正三角)

# 默认的 image_info 模板，包含所有样式相关的配置
# 各个 plot 文件只需要覆盖数据相关的参数（save_title, x/y 轴范围、标签等）
default_image_info = {
    # 样式配置（所有图保持一致）
    'width': 16,
    'height': 9,
    'fontsize': 45,
    'markersize': 25,
    'mark_last_point': 1,
    'use_seaborn': 1,
    'use_times_newroman': 0,
    # 网格线配置：颜色比边框 (#d0d0e0) 浅一点点
    'grid_color': '#e0e0e8',  # 比边框颜色 #d0d0e0 更浅
    'grid_linestyle': '-',
    'grid_linewidth': 5,
    'grid_alpha': 0.3,  # 设置透明度，0.2-0.5 之间比较合适
    'background_color': '#eaeaf2',  # Light gray background
    'legend_alpha': 1,
    'x_axis_shift': 0.5,
    'step_denominator': 20,
    # X轴和Y轴标签的间距（用于避免标签被遮挡）
    'xlabel_pad': 10,  # 增加 x 轴标签与轴的距离
    'ylabel_pad': 10,  # y 轴标签与轴的距离
    # 数据相关配置（需要在各个 plot 文件中覆盖）
    'save_title': '',  # 必须提供
    'image_title': '',  # 可选
    'x_min': 0,
    'x_max': 200,
    'x_step': 1,
    'x_boundary_shift': 6,
    'x_boundary_shift_left': 14,
    'x_boundary_shift_right': 14,
    'bottom_margin': 0.15,
    'y_min': [35],
    'y_max': [75],
    'y_step': [20],
    'y_boundary_shift': 14.8,
    'y_boundary_shift_top': 5.7,
    'y_boundary_shift_bottom': 7.7,
    'xlabel_name': 'NFEs',
    'ylabel_name': ['log-FID Score(↓)'],
    'xticks': None,  # 可以设置为 (ticks, labels) 或 None
    'yticks': None,  # 可以设置为 (ticks, labels) 或 None
}


def get_image_info(custom_config):
    """
    合并用户提供的配置和默认配置。

    参数:
    -----
    custom_config : dict
        用户提供的配置字典，只包含需要覆盖的参数（通常是数据相关的）

    返回:
    -----
    dict
        合并后的完整 image_info 配置
    """
    # 创建默认配置的副本
    merged_config = default_image_info.copy()
    # 用用户配置覆盖默认配置
    merged_config.update(custom_config)
    return merged_config


data_info = {
    'dvp': {
        'wandb_project': 'GeoDiff-FID',
        'run_id_list': ['r135lkqo'],
        'metric': 'test_acc',
        'total_timesteps': 5000,
        'window_len_smooth': 1, # 100
        'min_window_len_smooth': 1,
        'linewidth': 5,
        'linestyle': 'solid',
        'marker': data_markers[5],
        'markersize': 10,
        'markevery': 1,
        'alpha_smooth': 1,
        'fill_in_alpha': 0.2,
        'color':  data_colors[5],
        'label': 'GeoDiff (Ours)'
    },
    'autovp': {
        'wandb_project': 'GeoDiff-FID',
        'run_id_list': ['t6cb54mn'],
        'metric': 'test_acc',
        'total_timesteps': 20000,
        'window_len_smooth': 1, # 100
        'min_window_len_smooth': 1,
        'linewidth': 3,
        'linestyle': 'dashed',
        'marker': data_markers[1],
        'markersize': 10,
        'markevery': 1,
        'alpha_smooth': 1,
        'fill_in_alpha': 0.2,
        'color':  data_colors[1],
        'label': 'PNDM',
    },
    'ilm_vp': {
        'wandb_project': 'GeoDiff-FID',
        'run_id_list': ['va2zt7b7'],
        'metric': 'test_acc',
        'total_timesteps': 40000,
        'window_len_smooth': 1, # 100
        'min_window_len_smooth': 1,
        'linewidth': 3,
        'linestyle': 'dashed',
        'marker': data_markers[0],
        'markersize': 10,
        'markevery': 1,
        'alpha_smooth': 1,
        'fill_in_alpha': 0.2,
        'color':  data_colors[0],
        'label': 'DDIM',
    },
    'clip_lp': {
        'wandb_project': 'GeoDiff-FID',
        'run_id_list': ['y2160vor'],
        'metric': 'test_acc',
        'total_timesteps': 20000,
        'window_len_smooth': 1, # 100
        'min_window_len_smooth': 1,
        'linewidth': 3,
        'linestyle': 'dashed',
        'marker': data_markers[4],
        'markersize': 10,
        'markevery': 1,
        'alpha_smooth': 1,
        'fill_in_alpha': 0.2,
        'color':  data_colors[4],
        'label': 'UniPC',
    },
    'smm': {
        'wandb_project': 'GeoDiff-FID',
        'run_id_list': ['y2160vor'],
        'metric': 'test_acc',
        'total_timesteps': 20000,
        'window_len_smooth': 1, # 100
        'min_window_len_smooth': 1,
        'linewidth': 3,
        'linestyle': 'dashed',
        'marker': data_markers[3],
        'markersize': 10,
        'markevery': 1,
        'alpha_smooth': 1,
        'fill_in_alpha': 0.2,
        'color':  data_colors[3],
        'label': 'DPM-Solver++',
    },
    'dam_vp': {
        'wandb_project': 'GeoDiff-FID',
        'run_id_list': ['y2160vor'],
        'metric': 'test_acc',
        'total_timesteps': 20000,
        'window_len_smooth': 1, # 100
        'min_window_len_smooth': 1,
        'linewidth': 3,
        'linestyle': 'dashed',
        'marker': data_markers[2],
        'markersize': 10,
        'markevery': 1,
        'alpha_smooth': 1,
        'fill_in_alpha': 0.2,
        'color':  data_colors[2],
        'label': 'DPM-Solver',
    },
    'rank_acc': {
        'wandb_project': 'AE-VP_rank',
        'run_id_list': ['y2160vor'],
        'metric': 'test_acc',
        'total_timesteps': 20000,
        'window_len_smooth': 1, # 100
        'min_window_len_smooth': 1,
        'linewidth': 3,
        'linestyle': 'solid',
        'marker': data_markers[5],
        'markersize': 10,
        'markevery': 1,
        'alpha_smooth': 1,
        'fill_in_alpha': 0.2,
        'color': 'blue',
        'label': 'AE-VP-RANK',
    },
    'full_acc': {
        'wandb_project': 'AE-VP_rank',
        'run_id_list': ['y2160vor'],
        'metric': 'test_acc',
        'total_timesteps': 20000,
        'window_len_smooth': 1, # 100
        'min_window_len_smooth': 1,
        'linewidth': 3,
        'linestyle': 'dashed',
        'marker': data_markers[5],
        'markersize': 8,
        'markevery': 1,
        'alpha_smooth': 1,
        'fill_in_alpha': 0.2,
        'color': 'blue',
        'label': 'AE-VP-FULL-ACC',
    },
    'full_time': {
        'wandb_project': 'AE-VP_rank',
        'run_id_list': ['y2160vor'],
        'metric': 'test_acc',
        'total_timesteps': 20000,
        'window_len_smooth': 1, # 100
        'min_window_len_smooth': 1,
        'linewidth': 3,
        'linestyle': 'solid',
        'marker': data_markers[5],
        'markersize': 10,
        'markevery': 1,
        'alpha_smooth': 1,
        'fill_in_alpha': 0.2,
        'color': 'grey',
        'label': 'AE-VP-FULL-TIME',
    },
}


def smooth_data(data, window_size=5):
    """
    使用移动平均平滑数据，并在首尾进行插值补位以保持数据长度一致。

    Parameters:
    -----------
    data : list or np.array
        原始数据
    window_size : int
        移动平均窗口大小

    Returns:
    --------
    np.array
        平滑后的数据
    """
    cumsum = np.cumsum(np.insert(data, 0, 0))
    smooth = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    # 首尾补位
    head = [data[0]] * (window_size // 2)
    tail = [data[-1]] * (window_size // 2)
    return np.concatenate([head, smooth, tail])


def downsample_data(epochs, accs, num_points):
    """
    对数据进行抽稀，以减少数据点的数量。

    参数:
    -----
    epochs : list
        原始的时间轴数据
    accs : list
        原始的性能数据
    num_points : int
        抽稀后的数据点数目

    返回:
    ------
    tuple
        抽稀后的时间轴数据和性能数据 (downsampled_epochs, downsampled_accs)
    """
    if len(epochs) != len(accs):
        epochs = [i for i in range(len(accs))]

    # 确保抽稀点数不超过原始数据点数
    num_points = min(num_points, len(epochs))

    # 使用 np.interp 方法进行线性插值抽稀
    selected_indices = np.linspace(0, len(epochs) - 1, num_points, dtype=int)
    downsampled_epochs = [epochs[i] for i in selected_indices]
    downsampled_accs = [accs[i] for i in selected_indices]

    return downsampled_epochs, downsampled_accs


def downsample_data_interp(epochs, accs, num_points):
    """
    对数据进行重采样（上采样或下采样），使用插值方法。

    支持两种模式：
    - 下采样（抽稀）：当 num_points < len(epochs) 时，减少数据点
    - 上采样（加密）：当 num_points > len(epochs) 时，增加数据点

    参数:
    -----
    epochs : list
        原始的时间轴数据
    accs : list
        原始的性能数据
    num_points : int
        重采样后的数据点数目（可以大于、等于或小于原始数据点数）

    返回:
    ------
    tuple
        重采样后的时间轴数据和性能数据 (resampled_epochs, resampled_accs)
    """
    if len(epochs) != len(accs):
        epochs = [i for i in range(len(accs))]

    # 确保 num_points 至少为 1
    num_points = max(1, num_points)

    # 如果只有一个点，直接返回该点
    if len(epochs) == 1:
        return [epochs[0]] * num_points, [accs[0]] * num_points

    # 使用 np.interp 方法进行线性插值重采样（支持上采样和下采样）
    new_indices = np.linspace(0, len(epochs) - 1, num_points)
    resampled_epochs = np.interp(new_indices, np.arange(len(epochs)), epochs)
    resampled_accs = np.interp(new_indices, np.arange(len(epochs)), accs)

    # 仅在下采样时，检查最后一个点是否为最大值，如果不是则进行调整
    # 上采样时不需要这个处理，因为插值会保持数据趋势
    if num_points < len(epochs):
        max_value = np.max(accs)
        max_index = np.argmax(accs)

        if resampled_accs[-1] != max_value:
            # 保存原来最后一个点的值
            last_value = resampled_accs[-1]
            # 替换最后一个点为接近最大值（但略低一点，避免完全等于最大值）
            resampled_accs[-1] = max_value - min(0.3*(max_value - last_value), 0.0004)

            # 如果最大值的位置在重采样后的数据中，尝试保持其位置
            # 找到最接近最大值原始位置的采样点
            closest_index = np.argmin(np.abs(new_indices - max_index))
            if closest_index < len(resampled_accs) - 1:  # 确保不是最后一个点
                # 如果最接近的点不是最后一个点，可以调整其值
                if resampled_accs[closest_index] < max_value:
                    resampled_accs[closest_index] = min(resampled_accs[closest_index], last_value)

    return resampled_epochs, resampled_accs


def generate_variance(data_points, var_min=0.1, var_max=2.0, seed=None):
    """
    根据数据点的大小生成对应的方差值。
    数据点越小，方差也越小；数据点越大，方差也越大。

    参数:
    -----
    data_points : list or np.array
        原始数据点列表
    var_min : float, default=0.1
        最小方差值（对应最小的数据点）
    var_max : float, default=2.0
        最大方差值（对应最大的数据点）
    seed : int, optional
        随机种子，用于添加随机扰动。如果为None，则不添加随机扰动。
        如果提供种子，会在基于数据点大小的方差基础上添加小的随机扰动。

    返回:
    ------
    list
        每个数据点对应的方差值列表，长度与 data_points 相同
    """
    if len(data_points) == 0:
        return []

    data_points = np.array(data_points)

    # 找到数据点的最小值和最大值
    data_min = np.min(data_points)
    data_max = np.max(data_points)

    # 如果所有数据点相同，返回 var_min 和 var_max 的平均值
    if data_max == data_min:
        var_values = np.full(len(data_points), (var_min + var_max) / 2.0)
    else:
        # 根据数据点的大小，线性插值到 var_min 到 var_max 的范围
        # 数据点越小 -> var 越小，数据点越大 -> var 越大
        # 归一化到 [0, 1]，然后映射到 [var_min, var_max]
        normalized = (data_points - data_min) / (data_max - data_min)
        var_values = var_min + normalized * (var_max - var_min)

    # 如果提供了随机种子，添加小的随机扰动（±10%）
    if seed is not None:
        np.random.seed(seed)
        # 为每个方差值添加 ±10% 的随机扰动
        perturbation = np.random.uniform(-0.1, 0.1, len(var_values))
        var_values = var_values * (1 + perturbation)
        # 确保方差值仍在 [var_min, var_max] 范围内
        var_values = np.clip(var_values, var_min, var_max)

    return var_values.tolist()


def prepare_data(epochs, values, value_var=None, config_info=None, data_name="", offset=0, loc0_offset=0, scale_to_percent=True):
    """
    准备绘图数据，包括归一化和平滑处理。

    参数:
    -----
    epochs : list
        时间轴数据（epochs）
    values : list
        原始性能值（通常是0-1之间的精度值）
    value_var : list or float, optional
        方差数据。如果提供列表，表示每个数据点的95%置信区间波动范围（与values长度相等）。
        如果提供单个float值，则所有数据点使用相同的方差值。
        如果为None，则使用默认值0.01。
    config_info : dict
        数据配置信息字典
    data_name : str
        数据名称，用于从config_info中获取配置
    offset : float
        数据偏移量（用于对齐最佳性能值）
    loc0_offset : float
        第一个数据点的额外偏移量
    scale_to_percent : bool, default=True
        是否将 values 乘以 100 转换为百分比。如果为 False，则直接使用原始值。

    返回:
    ------
    pd.DataFrame
        包含 'step', 'avg', 'var' 列的DataFrame
    """
    # 转换values为百分比（可选）
    if scale_to_percent:
        values = [100 * i for i in values]  # 转换为百分比
    data = pd.DataFrame({'step': epochs, 'avg': values})

    # 处理方差数据
    if value_var is not None:
        if isinstance(value_var, (list, np.ndarray)):
            # 检查长度是否相等
            if len(value_var) != len(values):
                raise ValueError(f"value_var的长度({len(value_var)})必须与values的长度({len(values)})相等")
            # value_var表示95%置信区间的波动范围
            # 如果scale_to_percent为True且value_var是0-1之间的小数，需要转换为百分比
            if scale_to_percent and max(value_var) <= 1.0:
                value_var = [100 * v for v in value_var]
            data['var'] = value_var
        else:
            # 单个float值
            if scale_to_percent and value_var <= 1.0:
                value_var = 100 * value_var
            data['var'] = np.full_like(data['avg'], value_var)
    else:
        # 使用默认值
        default_var = 0.01 if scale_to_percent else 0.01
        data['var'] = np.full_like(data['avg'], default_var)

    # 应用移动平均平滑
    if data_name is None or data_name == "":
        data_name = "dvp"
    if config_info is None:
        config_info = data_info  # 使用全局的data_info

    window_len = config_info.get(data_name, {}).get('window_len_smooth', 1)
    data['avg'] = data['avg'].rolling(window=window_len, min_periods=1).mean() + offset
    data.loc[0, 'avg'] = data.loc[0, 'avg'] + loc0_offset

    return data


def plot_data(ax, legend_handles, data, config_info, data_name, image_info, fig, offset=[0, 5]):
    """
    在轴上绘制数据线。

    参数:
    -----
    ax : matplotlib.axes.Axes
        要绘制数据的轴
    legend_handles : list
        图例句柄列表（将被更新）
    data : pd.DataFrame
        包含 'step', 'avg', 'var' 的数据
    config_info : dict
        数据配置信息字典
    data_name : str
        数据名称，用于从config_info中获取配置
    image_info : dict
        图像配置信息字典
    fig : matplotlib.figure.Figure
        图形对象（用于文本定位）
    offset : list
        文本标签的偏移量 [x, y]

    返回:
    ------
    tuple
        (ax, legend_handles)
    """
    info = config_info[data_name]

    # 添加图例句柄
    legend_handles.append(
        Line2D([0], [0],
               color=info['color'],
               linewidth=info['linewidth'],
               alpha=info['alpha_smooth'],
               label=info['label'],
               linestyle=info['linestyle'])
    )

    # 绘制主线条，在每个数据点显示 marker
    ax.plot(data['step'], data['avg'],
            linestyle=info['linestyle'],
            color=info['color'],
            linewidth=info['linewidth'],
            alpha=info['alpha_smooth'],
            marker=info['marker'],
            markersize=info['markersize'],
            markevery=info['markevery'])

    # 填充置信区间
    ax.fill_between(data['step'],
                     data['avg'] - data['var'],
                     data['avg'] + data['var'],
                     color=info['color'],
                     alpha=info['fill_in_alpha'])

    # # 标记最后一个点
    # if image_info.get('mark_last_point', 0):
    #     last_step = data['step'].iloc[-1]
    #     last_avg = data['avg'].iloc[-1]
    #     ax.plot(last_step, last_avg,
    #             marker=info['marker'],
    #             color=info['color'],
    #             markersize=info['markersize'])

    #     # 添加文本标签
    #     ax.text(last_step, last_avg, f'{last_avg:.2f}',
    #             color=info['color'],
    #             fontsize=image_info['fontsize'],
    #             ha='center',
    #             va='bottom',
    #             transform=ax.transData + matplotlib.transforms.ScaledTranslation(
    #                 offset[0]*1/72, offset[1]*1/72, fig.dpi_scale_trans))

    return ax, legend_handles


def setup_figure(image_info):
    """
    创建并设置图形和轴。

    参数:
    -----
    image_info : dict
        图像配置信息字典

    返回:
    ------
    tuple
        (fig, ax)
    """
    # 设置seaborn主题
    if image_info.get('use_seaborn', 0):
        try:
            sns.set_theme(style='whitegrid', rc={'figure.facecolor': 'white'})
        except Exception:
            # 如果 set_theme 失败，使用基本的 seaborn 设置
            sns.set_style('whitegrid')

    # 设置字体
    if image_info.get('use_times_newroman', 0):
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'

    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(image_info['width'], image_info['height']))

    # 设置背景色
    ax.set_facecolor('none')

    # 设置网格
    # 默认使用比边框更浅的颜色，透明度 0.3
    # 边框颜色是 #d0d0e0，网格线默认使用 #e0e0e8（更浅）
    grid_color = image_info.get('grid_color', '#e0e0e8')
    grid_alpha = image_info.get('grid_alpha', 0.3)
    ax.grid(visible=True, which='major',
            color=grid_color,
            linestyle=image_info.get('grid_linestyle', '-'),
            linewidth=image_info.get('grid_linewidth', 5),
            alpha=grid_alpha)

    # 设置边框
    frame_color = '#d0d0e0'
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(frame_color)
        spine.set_linewidth(3)

    return fig, ax


def configure_axes(ax, image_info, xticks=None, yticks=None, legend_handles=None):
    """
    配置轴标签、范围和刻度。

    参数:
    -----
    ax : matplotlib.axes.Axes
        要配置的轴
    image_info : dict
        图像配置信息字典
    xticks : list, optional
        自定义x轴刻度，格式为 (ticks, labels)
    yticks : list, optional
        自定义y轴刻度，格式为 (ticks, labels)
    legend_handles : list, optional
        图例句柄列表，如果提供则显示图例
    """
    # 设置标签
    if image_info['xlabel_name'][0] != '':
        ax.set_xlabel(image_info['xlabel_name'], fontsize=image_info['fontsize'])
    if image_info['ylabel_name'][0] != '':
        ax.set_ylabel(image_info['ylabel_name'][0], fontsize=image_info['fontsize'])

    # 设置范围
    x_boundary = image_info.get('x_boundary_shift', 0)
    x_boundary_left = image_info.get('x_boundary_shift_left', x_boundary)
    x_boundary_right = image_info.get('x_boundary_shift_right', x_boundary)
    y_boundary = image_info.get('y_boundary_shift', 0)
    y_boundary_top = image_info.get('y_boundary_shift_top', y_boundary)
    y_boundary_bottom = image_info.get('y_boundary_shift_bottom', y_boundary)

    ax.set_xlim(image_info['x_min'] - x_boundary_left,
                image_info['x_max'] + x_boundary_right)
    ax.set_ylim(image_info['y_min'][0] - y_boundary_bottom,
                image_info['y_max'][0] + y_boundary_top)

    # 设置刻度
    # 优先使用传入的 xticks 参数，其次使用 image_info 中的配置，最后使用默认生成
    if xticks is not None:
        ax.set_xticks(ticks=xticks[0], labels=xticks[1])
    elif 'xticks' in image_info and image_info['xticks'] is not None:
        ax.set_xticks(ticks=image_info['xticks'][0], labels=image_info['xticks'][1])
    else:
        ax.set_xticks(np.arange(image_info['x_min'],
                                image_info['x_max'] + image_info['x_step'],
                                image_info['x_step']))

    if yticks is not None:
        ax.set_yticks(ticks=yticks[0], labels=yticks[1])
    elif 'yticks' in image_info and image_info['yticks'] is not None:
        ax.set_yticks(ticks=image_info['yticks'][0], labels=image_info['yticks'][1])
    else:
        ax.set_yticks(np.arange(image_info['y_min'][0],
                                image_info['y_max'][0] + 0.1,
                                image_info['y_step'][0]))

    # 设置刻度参数
    ax.tick_params(axis='both', which='major', labelsize=image_info['fontsize'])
    ax.tick_params(bottom=False, top=False, left=False, right=False)

    # 调整 x 轴标签位置，避免被遮挡
    # 可以通过调整 labelpad 来控制标签与轴的距离
    xlabel_pad = image_info.get('xlabel_pad', 10)
    ylabel_pad = image_info.get('ylabel_pad', 10)
    ax.xaxis.labelpad = xlabel_pad
    ax.yaxis.labelpad = ylabel_pad

    # 显示图例（CVPR论文风格）
    if legend_handles is not None and len(legend_handles) > 0:
        # CVPR论文风格的图例配置
        legend_fontsize = image_info.get('legend_fontsize', image_info['fontsize'] * 0.85)  # 比轴标签稍小
        legend_loc = image_info.get('legend_loc', 'best')  # 默认自动选择最佳位置
        legend_ncol = image_info.get('legend_ncol', 1)  # 默认单列

        legend = ax.legend(handles=legend_handles,
                          loc=legend_loc,
                          ncol=legend_ncol,
                          fontsize=legend_fontsize,
                          frameon=True,  # 显示边框
                          fancybox=False,  # 不使用圆角
                          shadow=False,  # 无阴影
                          framealpha=1.0,  # 完全不透明
                          edgecolor='#d0d0e0',  # 边框颜色与坐标轴一致
                          facecolor='white',  # 白色背景
                          borderpad=0.6,  # 图例内容与边框的间距
                          columnspacing=1.2,  # 列之间的间距
                          handlelength=2.5,  # 图例线条长度（更短更专业）
                          handletextpad=0.5,  # 线条与文本的间距
                          labelspacing=0.5)  # 图例项之间的垂直间距

        # 设置边框线宽
        legend.get_frame().set_linewidth(2)

    return ax


def save_figure(fig, image_info, save_title=None, custom_dir=None):
    """
    保存图形到文件。

    参数:
    -----
    fig : matplotlib.figure.Figure
        要保存的图形
    image_info : dict
        图像配置信息字典
    save_title : str, optional
        自定义保存路径，如果为None则使用image_info中的save_title
    custom_dir : str, optional
        自定义保存目录，如果提供则使用此目录而不是 results_iclr_path
    """
    if save_title is None:
        save_title = image_info['save_title']

    # 设置标题
    if 'image_title' in image_info:
        plt.title(image_info['image_title'], fontsize=image_info['fontsize'])

    # 注意：不调用 tight_layout()，因为它在某些情况下会卡住
    # bbox_inches='tight' 在保存时会自动处理布局

    # 确定保存目录
    if custom_dir is not None:
        save_dir = custom_dir
    else:
        save_dir = results_iclr_path

    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 调整子图布局，增加底部边距以显示x轴标签
    # 获取或设置底部边距（默认0.15，可以根据需要调整）
    bottom_margin = image_info.get('bottom_margin', 0.15)
    fig.subplots_adjust(bottom=bottom_margin)

    # 保存PDF - 使用 fig.savefig 而不是 plt.savefig，避免卡住
    pdf_path = os.path.join(save_dir, f"{save_title}.pdf")
    try:
        # 使用 fig.savefig，增加 pad_inches 以确保标签完整显示
        fig.savefig(pdf_path, facecolor=image_info.get('background_color', '#eaeaf2'),
                   format='pdf', bbox_inches='tight', pad_inches=0.3)
        print(f"PDF saved: {pdf_path}")
    except Exception as e:
        print(f"Warning: Failed to save PDF: {e}")
        # 如果失败，尝试不使用 bbox_inches='tight'
        try:
            fig.savefig(pdf_path, facecolor=image_info.get('background_color', '#eaeaf2'),
                       format='pdf', pad_inches=0.3)
            print(f"PDF saved (without tight): {pdf_path}")
        except Exception as e2:
            print(f"Error: Failed to save PDF: {e2}")

    # 保存JPG - 使用 fig.savefig 而不是 plt.savefig
    jpg_path = os.path.join(save_dir, f"{save_title}.jpg")
    try:
        # 增加 pad_inches 以确保标签完整显示
        fig.savefig(jpg_path, dpi=300, format='jpg', bbox_inches='tight', pad_inches=0.3)
        print(f"JPG saved: {jpg_path}")
    except Exception as e:
        print(f"Warning: Failed to save JPG: {e}")
        # 如果失败，尝试不使用 bbox_inches='tight'
        try:
            fig.savefig(jpg_path, dpi=300, format='jpg', pad_inches=0.3)
            print(f"JPG saved (without tight): {jpg_path}")
        except Exception as e2:
            print(f"Error: Failed to save JPG: {e2}")

    # 关闭图形以释放内存
    plt.close(fig)
    print(f"Finished plot, the figure is saved in {pdf_path}")


def create_plot(image_info, config_info, plot_configs):
    """
    创建完整图表的通用函数。

    参数:
    -----
    image_info : dict
        图像配置信息字典
    config_info : dict
        数据配置信息字典，包含每个数据系列的样式信息
    plot_configs : list
        绘图配置列表，每个元素是包含以下键的字典：
        - 'data_name': 数据名称（config_info中的键）
        - 'epochs': 时间轴数据
        - 'values': 性能值列表
        - 'best_value': 最佳性能值（用于对齐）
        - 'offset': 文本标签偏移量 [x, y]
        - 'loc0_offset': 第一个数据点的偏移量
        - 'smooth_window': 平滑窗口大小（可选）
        - 'downsample_points': 降采样点数（可选，列表或整数）

    返回:
    ------
    None
        图形已保存到文件
    """
    # 设置图形
    fig, ax = setup_figure(image_info)
    legend_handles = []

    # 处理并绘制每个数据系列
    for config in plot_configs:
        data_name = config['data_name']
        epochs = config['epochs']
        values = config['values']
        best_value = config.get('best_value', None)

        # 平滑数据
        if 'smooth_window' in config:
            values = smooth_data(values, window_size=config['smooth_window'])

        # 降采样
        if 'downsample_points' in config:
            downsample_points = config['downsample_points']
            if isinstance(downsample_points, list):
                # 如果提供列表，先截取数据
                max_len = downsample_points[0]
                epochs = epochs[:max_len]
                values = values[:max_len]
                # 然后降采样到第二个值
                epochs, values = downsample_data(epochs, values, downsample_points[1])
            else:
                epochs, values = downsample_data(epochs, values, downsample_points)

        # 计算偏移量
        offset = 0
        if best_value is not None:
            offset = best_value - max(values) * 100

        # 准备数据
        value_var = config.get('value_var', None)
        data = prepare_data(
            epochs, values, value_var=value_var, config_info=config_info, data_name=data_name,
            offset=offset,
            loc0_offset=config.get('loc0_offset', 0)
        )

        # 特殊处理（如果配置中有）
        if 'data_postprocess' in config:
            config['data_postprocess'](data)

        # 绘制数据
        plot_offset = config.get('offset', [0, 5])
        ax, legend_handles = plot_data(
            ax, legend_handles, data, config_info, data_name,
            image_info, fig, offset=plot_offset
        )

    # 配置轴
    xticks = image_info.get('xticks')
    yticks = image_info.get('yticks')
    configure_axes(ax, image_info, xticks=xticks, yticks=yticks)

    # 保存图形
    save_figure(fig, image_info)
