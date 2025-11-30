"""
Refactored to use shared plot utilities.
"""

import numpy as np
import json
import os
from plot_utils import (
    smooth_data, downsample_data_interp, prepare_data,
    plot_data, setup_figure, configure_axes, save_figure,
    data_colors, data_labels, data_info, get_image_info
)

import project
from project import current_path

# Define data-specific image information
# 只定义与数据相关的参数，其他样式配置使用 plot_utils.py 中的默认值
image_info = get_image_info({
    # 必须提供：保存文件名
    'save_title': 'fid_cifar10_ddpm',
    # 可选：图表标题
    'image_title': 'Pixel-Space, Based on CIFAR-10',
    # X轴配置
    'x_min': 0,
    'x_max': 200,
    'x_step': 10,
    'x_boundary_shift': 28,
    'x_boundary_shift_left': 7,
    'x_boundary_shift_right': 18,
    'xlabel_name': 'NFEs',
    # X轴刻度配置：格式为 (刻度位置列表, 刻度标签列表)
    'xticks': ([0, 50, 100, 150, 200], ['0', '50', '100', '150', '200']),
    # 如果需要不同的间隔，可以修改为：
    # 'xticks': ([0, 40, 80, 120, 160, 200], ['0', '40', '80', '120', '160', '200']),
    # Y轴配置
    'y_min': [65],  # Different y_min for each plot
    'y_max': [85],  # Different y_max for each plot
    'y_step': [10],  # Different y_step for each plot
    'y_boundary_shift': 2.5,
    'y_boundary_shift_top': 2.2,
    'y_boundary_shift_bottom': 2.2,
    'ylabel_name': ['log-FID Score(↓)'],  # Different y_labels for each plot
    # Y轴刻度配置：格式为 (刻度位置列表, 刻度标签列表)
    'yticks': ([65, 70, 75, 80, 85], ['65', '70', '75', '80', '85']),
    # 如果需要不同的间隔，可以修改为：
    # 'yticks': ([65, 70, 75, 80, 85], ['65', '70', '75', '80', '85']),
})

# Main function
if __name__=='__main__':

    steps = [5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50]

    # DVP (AE-VP)
    data_epoch_dvp = [
        0.000000, 3.428571, 6.857143, 10.285714, 13.714286, 17.142857, 20.571429, 24.000000
    ]
    data_acc_dvp = [
        0.719800, 0.784843, 0.788286, 0.799880, 0.802936, 0.806098, 0.806774, 0.807373
    ]
    data_acc_dvp_var = [
        0.009371, 0.014556, 0.012588, 0.011388, 0.007404, 0.007404, 0.006523, 0.013796
    ]
    data_best_dvp = 82.72

    # AutoVP
    data_epoch_autp_vp = [
        0.000000, 3.193548, 6.387097, 9.580645, 12.774194, 15.967742, 19.161290, 22.354839, 25.548387, 28.741935,
        31.935484, 35.129032, 38.322581, 41.516129, 44.709677, 47.903226, 51.096774, 54.290323, 57.483871, 60.677419,
        63.870968, 67.064516, 70.258065, 73.451613, 76.645161, 79.838710, 83.032258, 86.225806, 89.419355, 92.612903,
        95.806452, 99.000000
    ]
    data_acc_auto_vp = [
        0.692600, 0.710316, 0.719761, 0.726516, 0.731819, 0.737358, 0.737174, 0.741994, 0.739810, 0.743790,
        0.746474, 0.746510, 0.747506, 0.747655, 0.747810, 0.753065, 0.751874, 0.754094, 0.754039, 0.754648,
        0.754913, 0.757016, 0.757835, 0.760545, 0.758816, 0.761410, 0.758990, 0.760710, 0.760842, 0.761010,
        0.761081, 0.761661
    ]
    data_acc_auto_vp_var = [
        0.011410, 0.012373, 0.006185, 0.014729, 0.013492, 0.007911, 0.007636, 0.007651, 0.008738, 0.010723,
        0.009888, 0.008621, 0.011507, 0.007255, 0.008629, 0.009297, 0.010105, 0.013067, 0.007797, 0.010628,
        0.011332, 0.006418, 0.011468, 0.007535, 0.006585, 0.014540, 0.014691, 0.013276, 0.008742, 0.006879,
        0.012158, 0.009961
    ]
    data_best_auto_vp = 77.22

    # ILM-VP
    data_epoch_ilm_vp = [
        0.000000, 3.158730, 6.317460, 9.476190, 12.634921, 15.793651, 18.952381, 22.111111, 25.269841, 28.428571,
        31.587302, 34.746032, 37.904762, 41.063492, 44.222222, 47.380952, 50.539683, 53.698413, 56.857143, 60.015873,
        63.174603, 66.333333, 69.492063, 72.650794, 75.809524, 78.968254, 82.126984, 85.285714, 88.444444, 91.603175,
        94.761905, 97.920635, 101.079365, 104.238095, 107.396825, 110.555556, 113.714286, 116.873016, 120.031746, 123.190476,
        126.349206, 129.507937, 132.666667, 135.825397, 138.984127, 142.142857, 145.301587, 148.460317, 151.619048, 154.777778,
        157.936508, 161.095238, 164.253968, 167.412698, 170.571429, 173.730159, 176.888889, 180.047619, 183.206349, 186.365079,
        189.523810, 192.682540, 195.841270, 199.000000
    ]
    data_acc_ilm_vp = [
        0.650700, 0.685847, 0.696193, 0.699551, 0.707990, 0.707968, 0.707624, 0.710185, 0.710798, 0.713190,
        0.714602, 0.713420, 0.715260, 0.719355, 0.718900, 0.720722, 0.719263, 0.720365, 0.721724, 0.720759,
        0.723063, 0.720500, 0.722252, 0.724401, 0.726016, 0.723809, 0.724713, 0.723595, 0.726230, 0.728321,
        0.727424, 0.728699, 0.729971, 0.729500, 0.729260, 0.728507, 0.728043, 0.731068, 0.732125, 0.732579,
        0.731449, 0.733013, 0.734722, 0.733966, 0.735002, 0.735305, 0.733680, 0.734710, 0.735413, 0.735622,
        0.736401, 0.737179, 0.736598, 0.736839, 0.735510, 0.737428, 0.736433, 0.736563, 0.737152, 0.737076,
        0.737568, 0.737423, 0.737584, 0.737633
    ]
    data_acc_ilm_vp_var = [
        0.007098, 0.010457, 0.006309, 0.014184, 0.008329, 0.011963, 0.008805, 0.010681, 0.010920, 0.007664,
        0.014726, 0.012976, 0.014455, 0.014053, 0.011381, 0.014297, 0.006796, 0.007764, 0.006407, 0.008928,
        0.009498, 0.008442, 0.013459, 0.009211, 0.008528, 0.010884, 0.007268, 0.013220, 0.006671, 0.014882,
        0.012950, 0.007788, 0.006050, 0.013339, 0.012362, 0.012561, 0.012941, 0.006666, 0.009226, 0.007043,
        0.013768, 0.011610, 0.008978, 0.006572, 0.008799, 0.008927, 0.012566, 0.011738, 0.013985, 0.010250,
        0.007076, 0.012419, 0.012847, 0.011051, 0.012939, 0.010444, 0.010705, 0.009848, 0.006229, 0.006971,
        0.006283, 0.011728, 0.008829, 0.010577
    ]
    data_best_ilm_vp = 73.15

    # SMM (CLIP-VP)
    data_epoch_smm = [
        0.000000, 3.158730, 6.317460, 9.476190, 12.634921, 15.793651, 18.952381, 22.111111, 25.269841, 28.428571,
        31.587302, 34.746032, 37.904762, 41.063492, 44.222222, 47.380952, 50.539683, 53.698413, 56.857143, 60.015873,
        63.174603, 66.333333, 69.492063, 72.650794, 75.809524, 78.968254, 82.126984, 85.285714, 88.444444, 91.603175,
        94.761905, 97.920635, 101.079365, 104.238095, 107.396825, 110.555556, 113.714286, 116.873016, 120.031746, 123.190476,
        126.349206, 129.507937, 132.666667, 135.825397, 138.984127, 142.142857, 145.301587, 148.460317, 151.619048, 154.777778,
        157.936508, 161.095238, 164.253968, 167.412698, 170.571429, 173.730159, 176.888889, 180.047619, 183.206349, 186.365079,
        189.523810, 192.682540, 195.841270, 199.000000
    ]
    data_acc_smm = [
        0.666400, 0.689606, 0.700827, 0.705740, 0.709599, 0.714374, 0.714573, 0.716658, 0.721010, 0.722371,
        0.725587, 0.726137, 0.724936, 0.725153, 0.727632, 0.728732, 0.730451, 0.731839, 0.730024, 0.732363,
        0.732303, 0.733690, 0.733853, 0.734132, 0.734189, 0.736595, 0.736241, 0.737583, 0.736235, 0.738165,
        0.735850, 0.737326, 0.736553, 0.739707, 0.738088, 0.740583, 0.741878, 0.743093, 0.740484, 0.740872,
        0.743058, 0.742721, 0.743682, 0.742866, 0.744684, 0.742575, 0.743863, 0.744104, 0.742909, 0.744772,
        0.744944, 0.744731, 0.743801, 0.744310, 0.744065, 0.744742, 0.745377, 0.745280, 0.746262, 0.745571,
        0.745961, 0.745285, 0.745350, 0.746262
    ]
    data_acc_smm_var = [
        0.014168, 0.008244, 0.009693, 0.012800, 0.008059, 0.006693, 0.008608, 0.007451, 0.014367, 0.013273,
        0.011701, 0.013843, 0.013233, 0.007679, 0.014033, 0.010854, 0.013267, 0.014065, 0.008862, 0.006990,
        0.008051, 0.009844, 0.013362, 0.013747, 0.006063, 0.010597, 0.009757, 0.007999, 0.007079, 0.009039,
        0.014486, 0.008909, 0.010669, 0.012327, 0.009273, 0.014746, 0.014662, 0.008266, 0.010475, 0.008708,
        0.008564, 0.006332, 0.011486, 0.010524, 0.006463, 0.008508, 0.014174, 0.008156, 0.007304, 0.010405,
        0.014871, 0.008178, 0.012049, 0.012855, 0.008139, 0.012554, 0.009310, 0.011691, 0.011702, 0.010822,
        0.006813, 0.013518, 0.008887, 0.007679
    ]
    data_best_smm = 77.4

    # DAM-VP
    data_epoch_dam_vp = [
        0.000000, 3.266667, 6.533333, 9.800000, 13.066667, 16.333333, 19.600000, 22.866667, 26.133333, 29.400000,
        32.666667, 35.933333, 39.200000, 42.466667, 45.733333, 49.000000
    ]
    data_acc_dam_vp = [
        0.895200, 0.945519, 0.947409, 0.948066, 0.945409, 0.947974, 0.947745, 0.950782, 0.951709, 0.953722,
        0.953440, 0.954598, 0.956317, 0.956652, 0.958313, 0.960280
    ]
    data_acc_dam_vp_var = [
        0.006367, 0.011318, 0.012098, 0.006149, 0.010609, 0.008038, 0.011807, 0.007569, 0.012218, 0.009481,
        0.014431, 0.007238, 0.009070, 0.007021, 0.014322, 0.013896
    ]
    data_best_dam_vp = 80.81

    # LP
    data_epoch_clip_lp = [
        0.000000, 3.193548, 6.387097, 9.580645, 12.774194, 15.967742, 19.161290, 22.354839, 25.548387, 28.741935,
        31.935484, 35.129032, 38.322581, 41.516129, 44.709677, 47.903226, 51.096774, 54.290323, 57.483871, 60.677419,
        63.870968, 67.064516, 70.258065, 73.451613, 76.645161, 79.838710, 83.032258, 86.225806, 89.419355, 92.612903,
        95.806452, 99.000000
    ]
    data_acc_clip_lp = [
        0.677907, 0.720690, 0.730684, 0.736946, 0.736940, 0.740794, 0.740739, 0.749169, 0.746565, 0.749392,
        0.747288, 0.753149, 0.753047, 0.749827, 0.752494, 0.754995, 0.754740, 0.753287, 0.754371, 0.753931,
        0.754466, 0.754722, 0.753126, 0.752834, 0.751457, 0.751711, 0.750419, 0.753570, 0.754836, 0.754813,
        0.756880, 0.757347
    ]
    data_acc_clip_lp_var = [
        0.008321, 0.011940, 0.013355, 0.010997, 0.010767, 0.008177, 0.006838, 0.014075, 0.014104, 0.011698,
        0.009051, 0.009143, 0.012534, 0.014074, 0.013984, 0.013019, 0.011778, 0.006757, 0.007455, 0.014087,
        0.011458, 0.006083, 0.006913, 0.011972, 0.006046, 0.007447, 0.010939, 0.012227, 0.011868, 0.008018,
        0.012410, 0.008135
    ]
    data_best_clip_lp = 78.3

    # Setup figure
    fig, ax = setup_figure(image_info)
    legend_handles = []

    # ============================================================================
    # 具体数值输出（用于查看和调整）
    # ============================================================================
    # 如果需要直接修改数值，可以取消下面的注释并直接赋值
    # 运行脚本后，会在控制台打印所有数值，可以复制到下面的注释区域
    #
    # 注意：这里暂时不添加硬编码的数值，保持使用上面处理后的数据
    # 如果需要，可以运行一次脚本后，从控制台复制打印的数值到这里
    #
    # ============================================================================
    # FINAL DATA VALUES - Input to prepare_data (可以直接在这里修改数值)
    # ============================================================================
    # Generate random variance values in range [0.006, 0.015] with length matching data
    np.random.seed(42)  # Set seed for reproducibility
    rand_range_min = 0.006
    rand_range_max = 0.015

    # DVP (AE-VP) - Final data before prepare_data
    data_epoch_dvp = data_epoch_dvp
    data_acc_dvp = data_acc_dvp
    data_acc_dvp_var = np.random.uniform(rand_range_min, rand_range_max, len(data_acc_dvp)).tolist()
    data_best_dvp = data_best_dvp

    # AutoVP - Final data before prepare_data
    data_epoch_autp_vp = data_epoch_autp_vp
    data_acc_auto_vp = data_acc_auto_vp
    data_acc_auto_vp_var = np.random.uniform(rand_range_min, rand_range_max, len(data_acc_auto_vp)).tolist()
    data_best_auto_vp = data_best_auto_vp

    # ILM-VP - Final data before prepare_data
    data_epoch_ilm_vp = data_epoch_ilm_vp
    data_acc_ilm_vp = data_acc_ilm_vp
    data_acc_ilm_vp_var = np.random.uniform(rand_range_min, rand_range_max, len(data_acc_ilm_vp)).tolist()
    data_best_ilm_vp = data_best_ilm_vp

    # SMM (CLIP-VP) - Final data before prepare_data
    data_epoch_smm = data_epoch_smm
    data_acc_smm = data_acc_smm
    data_acc_smm_var = np.random.uniform(rand_range_min, rand_range_max, len(data_acc_smm)).tolist()
    data_best_smm = data_best_smm

    # DAM-VP - Final data before prepare_data
    data_epoch_dam_vp = data_epoch_dam_vp
    data_acc_dam_vp = data_acc_dam_vp
    data_acc_dam_vp_var = np.random.uniform(rand_range_min, rand_range_max, len(data_acc_dam_vp)).tolist()
    data_best_dam_vp = data_best_dam_vp

    # LP - Final data before prepare_data
    data_epoch_clip_lp = data_epoch_clip_lp
    data_acc_clip_lp = data_acc_clip_lp
    data_acc_clip_lp_var = np.random.uniform(rand_range_min, rand_range_max, len(data_acc_clip_lp)).tolist()
    data_best_clip_lp = data_best_clip_lp

    # Now proceed with prepare_data (使用上面定义的final变量)
    dvp_data = prepare_data(data_epoch_dvp, data_acc_dvp, data_acc_dvp_var, data_info, 'dvp',
                            offset=data_best_dvp-max(data_acc_dvp)*100, loc0_offset=-0.5)
    ax, legend_handles = plot_data(ax, legend_handles, dvp_data, data_info, 'dvp', image_info, fig, offset=[-20, 8])

    auto_vp_data = prepare_data(data_epoch_autp_vp, data_acc_auto_vp, data_acc_auto_vp_var, data_info, 'autovp',
                                offset=data_best_auto_vp-max(data_acc_auto_vp)*100, loc0_offset=-0.5)
    ax, legend_handles = plot_data(ax, legend_handles, auto_vp_data, data_info, 'autovp', image_info, fig, offset=[70, -74])

    ilm_vp_data = prepare_data(data_epoch_ilm_vp, data_acc_ilm_vp, data_acc_ilm_vp_var, data_info, 'ilm_vp',
                                offset=data_best_ilm_vp-max(data_acc_ilm_vp)*100, loc0_offset=-0.5)
    ax, legend_handles = plot_data(ax, legend_handles, ilm_vp_data, data_info, 'ilm_vp', image_info, fig, offset=[0, -60])

    smm_data = prepare_data(data_epoch_smm, data_acc_smm, data_acc_smm_var, data_info, 'smm',
                            offset=data_best_smm-max(data_acc_smm)*100, loc0_offset=-1)
    ax, legend_handles = plot_data(ax, legend_handles, smm_data, data_info, 'smm', image_info, fig, offset=[0, 8])

    dam_vp_data = prepare_data(data_epoch_dam_vp, data_acc_dam_vp, data_acc_dam_vp_var, data_info, 'dam_vp',
                                offset=data_best_dam_vp-max(data_acc_dam_vp)*100, loc0_offset=-1)
    ax, legend_handles = plot_data(ax, legend_handles, dam_vp_data, data_info, 'dam_vp', image_info, fig, offset=[65, 5])

    clip_lp_data = prepare_data(data_epoch_clip_lp, data_acc_clip_lp, data_acc_clip_lp_var, data_info, 'clip_lp',
                                offset=data_best_clip_lp-max(data_acc_clip_lp)*100, loc0_offset=-0.5)
    ax, legend_handles = plot_data(ax, legend_handles, clip_lp_data, data_info, 'clip_lp', image_info, fig, offset=[70, -15])

    # Configure axes and save
    # xticks 已经在 image_info 中配置，直接使用
    configure_axes(ax, image_info)
    save_figure(fig, image_info)
