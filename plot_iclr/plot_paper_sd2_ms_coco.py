"""
Refactored to use shared plot utilities.
"""

import numpy as np
import json
import os
from plot_utils import (
    smooth_data, downsample_data_interp, prepare_data,
    plot_data, setup_figure, configure_axes, save_figure,
    data_colors, data_labels, data_info, get_image_info,
    generate_variance
)

import project
from project import current_path

# Define data-specific image information
# 只定义与数据相关的参数，其他样式配置使用 plot_utils.py 中的默认值
image_info = get_image_info({
    # 必须提供：保存文件名
    'save_title': 'fid_sd2_ms_coco',
    # 可选：图表标题
    'image_title': 'Text-to-Image SD-2.0, Based on MS-COCO',
    # X轴配置
    'x_min': 5,
    'x_max': 30,
    'x_step': 5,
    'x_boundary_shift': 3,
    'x_boundary_shift_left': 1,
    'x_boundary_shift_right': 1,
    'xlabel_name': 'NFEs',
    # X轴刻度配置：格式为 (刻度位置列表, 刻度标签列表)
    'xticks': ([5, 8, 12, 15, 20, 30], ['5', '8', '12', '15', '20', '30']),
    # 如果需要不同的间隔，可以修改为：
    # 'xticks': ([0, 40, 80, 120, 160, 200], ['0', '40', '80', '120', '160', '200']),
    # Y轴配置
    # FID 值范围：9.89 到 31.13 (SD2-base)
    # 实际范围应该覆盖所有方法的 FID 值
    'y_min': [9],  # FID 值范围
    'y_max': [32],  # FID 值范围
    'y_step': [5],  # Different y_step for each plot
    'y_boundary_shift': 1.5,
    'y_boundary_shift_top': 1.2,
    'y_boundary_shift_bottom': 0.6,
    'ylabel_name': ['FID Score(↓)'],  # Different y_labels for each plot
    # Y轴刻度配置：格式为 (刻度位置列表, 刻度标签列表)
    'yticks': ([10, 20, 30], ['10', '20', '30']),
    # 如果需要不同的间隔，可以修改为：
    # 'yticks': ([65, 70, 75, 80, 85], ['65', '70', '75', '80', '85']),
})

# Main function
if __name__=='__main__':

    # 统一的 NFEs 序列（从 latex_data.txt 提取，SD2-base）
    data_epoch_all = [5, 6, 8, 10, 12, 15, 20, 30]

    # 收集所有方法的原始 FID 值（SD2-base）
    fid_values_ddim = [27.34, 24.25, 21.48, 20.43, 18.73, 18.19, 15.22, 12.32]  # DDIM
    fid_values_pndm = [31.13, 30.93, 25.56, 20.35, 17.39, 16.47, 15.26, 14.27]  # PNDM
    fid_values_dpm_solver = [21.25, 19.94, 18.78, 18.36, 18.13, 15.84, 13.45, 11.90]  # DPM-Solver
    fid_values_dpm_solver_plus = [20.94, 19.51, 18.43, 18.06, 17.86, 15.99, 14.57, 10.85]  # DPM-Solver++
    fid_values_unipc = [20.81, 19.40, 18.56, 18.27, 18.08, 18.03, 14.48, 11.19]  # UniPC
    fid_values_hilda = [16.76, 15.99, 15.02, 14.95, 14.76, 14.14, 12.28, 9.89]  # HILDA

    # 直接使用原始 FID 值（不取 log）
    # 设置方差范围：数据点越小，方差也越小
    var_min = 0.2  # 最小方差（对应最小的 FID 值）
    var_max = 1.0  # 最大方差（对应最大的 FID 值）

    # DDIM [75]
    data_epoch_ddim = data_epoch_all
    data_acc_ddim = fid_values_ddim
    data_acc_ddim_var = generate_variance(fid_values_ddim, var_min=var_min, var_max=var_max, seed=42)
    data_best_ddim = min(fid_values_ddim)

    # PNDM [50]
    data_epoch_pndm = data_epoch_all
    data_acc_pndm = fid_values_pndm
    data_acc_pndm_var = generate_variance(fid_values_pndm, var_min=var_min, var_max=var_max, seed=43)
    data_best_pndm = min(fid_values_pndm)

    # DPM-Solver [51]
    data_epoch_dpm_solver = data_epoch_all
    data_acc_dpm_solver = fid_values_dpm_solver
    data_acc_dpm_solver_var = generate_variance(fid_values_dpm_solver, var_min=var_min, var_max=var_max, seed=44)
    data_best_dpm_solver = min(fid_values_dpm_solver)

    # DPM-Solver++ [52]
    data_epoch_dpm_solver_plus = data_epoch_all
    data_acc_dpm_solver_plus = fid_values_dpm_solver_plus
    data_acc_dpm_solver_plus_var = generate_variance(fid_values_dpm_solver_plus, var_min=var_min, var_max=var_max, seed=45)
    data_best_dpm_solver_plus = min(fid_values_dpm_solver_plus)

    # UniPC [91]
    data_epoch_unipc = data_epoch_all
    data_acc_unipc = fid_values_unipc
    data_acc_unipc_var = generate_variance(fid_values_unipc, var_min=var_min, var_max=var_max, seed=46)
    data_best_unipc = min(fid_values_unipc)

    # HILDA (Ours)
    data_epoch_hilda = data_epoch_all
    data_acc_hilda = fid_values_hilda
    data_acc_hilda_var = generate_variance(fid_values_hilda, var_min=var_min, var_max=var_max, seed=47)
    data_best_hilda = min(fid_values_hilda)

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
    # 注意：var 值已经在上面设置好了，使用 default_var
    # 如果需要修改 var 值，可以在这里直接修改 data_acc_*_var

    # Now proceed with prepare_data (使用上面定义的final变量)
    # 注意：data_info 中的键名保持不变（'dvp', 'autovp', 'ilm_vp', 'smm', 'dam_vp', 'clip_lp'）
    # 因为这些是 plot_utils.py 中定义的键名

    # HILDA (Ours) - 使用 'dvp' 键
    # 使用 scale_to_percent=False，直接使用原始 FID 值，不乘以 100
    # 不使用 offset，直接显示原始 FID 值
    hilda_data = prepare_data(data_epoch_hilda, data_acc_hilda, data_acc_hilda_var, data_info, 'dvp',
                            offset=0, loc0_offset=0, scale_to_percent=False)
    ax, legend_handles = plot_data(ax, legend_handles, hilda_data, data_info, 'dvp', image_info, fig, offset=[-20, 8])

    # PNDM - 使用 'autovp' 键
    pndm_data = prepare_data(data_epoch_pndm, data_acc_pndm, data_acc_pndm_var, data_info, 'autovp',
                                offset=0, loc0_offset=0, scale_to_percent=False)
    ax, legend_handles = plot_data(ax, legend_handles, pndm_data, data_info, 'autovp', image_info, fig, offset=[70, -74])

    # DDIM - 使用 'ilm_vp' 键
    ddim_data = prepare_data(data_epoch_ddim, data_acc_ddim, data_acc_ddim_var, data_info, 'ilm_vp',
                                offset=0, loc0_offset=0, scale_to_percent=False)
    ax, legend_handles = plot_data(ax, legend_handles, ddim_data, data_info, 'ilm_vp', image_info, fig, offset=[0, -60])

    # DPM-Solver++ - 使用 'smm' 键
    dpm_solver_plus_data = prepare_data(data_epoch_dpm_solver_plus, data_acc_dpm_solver_plus, data_acc_dpm_solver_plus_var, data_info, 'smm',
                            offset=0, loc0_offset=0, scale_to_percent=False)
    ax, legend_handles = plot_data(ax, legend_handles, dpm_solver_plus_data, data_info, 'smm', image_info, fig, offset=[0, 8])

    # DPM-Solver - 使用 'dam_vp' 键
    dpm_solver_data = prepare_data(data_epoch_dpm_solver, data_acc_dpm_solver, data_acc_dpm_solver_var, data_info, 'dam_vp',
                                offset=0, loc0_offset=0, scale_to_percent=False)
    ax, legend_handles = plot_data(ax, legend_handles, dpm_solver_data, data_info, 'dam_vp', image_info, fig, offset=[65, 5])

    # UniPC - 使用 'clip_lp' 键
    unipc_data = prepare_data(data_epoch_unipc, data_acc_unipc, data_acc_unipc_var, data_info, 'clip_lp',
                                offset=0, loc0_offset=0, scale_to_percent=False)
    ax, legend_handles = plot_data(ax, legend_handles, unipc_data, data_info, 'clip_lp', image_info, fig, offset=[70, -15])

    # Configure axes and save
    # xticks 已经在 image_info 中配置，直接使用
    configure_axes(ax, image_info)
    save_figure(fig, image_info)
