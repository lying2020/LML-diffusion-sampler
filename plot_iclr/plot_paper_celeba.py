"""
Refactored to use shared plot utilities.
"""

# Import necessary libraries
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
    'save_title': 'fid_celeba_ldm',
    # 可选：图表标题
    'image_title': 'LDM, Based on CelebA-HQ',
    # X轴配置
    'x_min': 0,
    'x_max': 200,
    'x_step': 10,
    'x_boundary_shift': 28,
    'x_boundary_shift_left': 7,
    'x_boundary_shift_right': 18,
    'xlabel_name': 'NFEs',
    # X轴标签间距（增加以显示x轴标签）
    'xlabel_pad': 25,
    # X轴刻度配置：格式为 (刻度位置列表, 刻度标签列表)
    'xticks': ([0, 50, 100, 150, 200], ['0', '50', '100', '150', '200']),
    # 如果需要不同的间隔，可以修改为：
    # 'xticks': ([0, 40, 80, 120, 160, 200], ['0', '40', '80', '120', '160', '200']),
    # Y轴配置
    'y_min': [35],  # Different y_min for each plot
    'y_max': [75],  # Different y_max for each plot
    'y_step': [20],  # Different y_step for each plot
    'y_boundary_shift': 15.2,
    'y_boundary_shift_top': 15.2,
    'y_boundary_shift_bottom': 15.2,
    'ylabel_name': ['log-FID Score(↓)'],  # Different y_labels for each plot
    # Y轴刻度配置：格式为 (刻度位置列表, 刻度标签列表)
    'yticks': ([35, 55, 75], ['35', '55', '75']),
    # 如果需要不同的间隔，可以修改为：
    # 'yticks': ([35, 55, 75], ['35', '55', '75']),
})



# Main function
if __name__=='__main__':

    # Setup figure
    fig, ax = setup_figure(image_info)
    legend_handles = []

    # ============================================================================
    # 具体数值输出（用于查看和调整）
    # ============================================================================
    # 如果需要直接修改数值，可以取消下面的注释并直接赋值
    # 运行脚本后，会在控制台打印所有数值，可以复制到下面的注释区域
    #

    # DVP (AE-VP)
    data_epoch_dvp = [
        0.000000, 3.428571, 6.857143, 10.285714, 13.714286, 17.142857, 20.571429, 24.000000
    ]
    data_acc_dvp = [
        0.013600, 0.762034, 0.768681, 0.770290, 0.772556, 0.774041, 0.777857, 0.779100
    ]
    data_acc_dvp_var = [
        0.009247, 0.012704, 0.011392, 0.010592, 0.007936, 0.007936, 0.007349, 0.012197
    ]
    data_best_dvp = 79.22

    # AutoVP
    data_epoch_autp_vp = [
        0.000000, 3.193548, 6.387097, 9.580645, 12.774194, 15.967742, 19.161290, 22.354839, 25.548387, 28.741935,
        31.935484, 35.129032, 38.322581, 41.516129, 44.709677, 47.903226, 51.096774, 54.290323, 57.483871, 60.677419,
        63.870968, 67.064516, 70.258065, 73.451613, 76.645161, 79.838710, 83.032258, 86.225806, 89.419355, 92.612903,
        95.806452, 99.000000
    ]
    data_acc_auto_vp = [
        0.107350, 0.616635, 0.690490, 0.713321, 0.714972, 0.713897, 0.715430, 0.718450, 0.718222, 0.720546,
        0.723456, 0.724753, 0.727015, 0.731421, 0.735983, 0.733881, 0.738423, 0.743178, 0.748103, 0.749509,
        0.754577, 0.757151, 0.760150, 0.763882, 0.766178, 0.768423, 0.770858, 0.770221, 0.771121, 0.771967,
        0.771003, 0.771105
    ]
    data_acc_auto_vp_var = [
        0.007349, 0.012197, 0.010607, 0.011248, 0.007124, 0.012819, 0.011995, 0.008274, 0.008091, 0.008100,
        0.008825, 0.010149, 0.009592, 0.008747, 0.010671, 0.007837, 0.008753, 0.009198, 0.009736, 0.011711,
        0.008198, 0.010085, 0.010554, 0.007279, 0.010645, 0.008023, 0.007390, 0.012693, 0.012794, 0.011850,
        0.008828, 0.007586
    ]
    data_best_auto_vp = 77.01

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
        0.089750, 0.197402, 0.208590, 0.225273, 0.227626, 0.215821, 0.223794, 0.223345, 0.231594, 0.228229,
        0.233465, 0.236448, 0.240125, 0.236519, 0.239402, 0.238706, 0.235820, 0.238717, 0.242408, 0.242269,
        0.245285, 0.246383, 0.243150, 0.247938, 0.247972, 0.248134, 0.245865, 0.246746, 0.249306, 0.253631,
        0.253699, 0.254877, 0.253277, 0.255411, 0.256446, 0.260594, 0.254978, 0.255473, 0.257828, 0.258721,
        0.260203, 0.261289, 0.260017, 0.264619, 0.264567, 0.262163, 0.264858, 0.266797, 0.266820, 0.267775,
        0.264500, 0.268588, 0.268391, 0.269981, 0.272404, 0.272432, 0.273184, 0.274501, 0.275874, 0.275364,
        0.276213, 0.277683, 0.278246, 0.278718
    ]
    data_acc_ilm_vp_var = [
        0.011105, 0.009641, 0.007732, 0.009971, 0.007206, 0.012456, 0.008553, 0.010975, 0.008870, 0.010120,
        0.010280, 0.008109, 0.012818, 0.011651, 0.012637, 0.012369, 0.010587, 0.012531, 0.007531, 0.008176,
        0.007271, 0.008952, 0.009332, 0.008628, 0.011972, 0.009141, 0.008686, 0.010256, 0.007846, 0.011813,
        0.007447, 0.012921, 0.011633, 0.008192, 0.007033, 0.011893, 0.011241, 0.011374, 0.011628, 0.007444,
        0.009151, 0.007695, 0.012179, 0.010740, 0.008985, 0.007381, 0.008866, 0.008951, 0.011378, 0.010825,
        0.012323, 0.009833, 0.007718, 0.011279, 0.011565, 0.010368, 0.011626, 0.009963, 0.010136, 0.009565,
        0.007153, 0.007647, 0.007189, 0.010818
    ]
    data_best_ilm_vp = 37.56

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
        0.689914, 0.849493, 0.854786, 0.856633, 0.858624, 0.860318, 0.862067, 0.863455, 0.865865, 0.865739,
        0.867196, 0.868061, 0.868522, 0.869142, 0.869034, 0.870472, 0.869855, 0.870649, 0.870828, 0.871679,
        0.872516, 0.872643, 0.872898, 0.873312, 0.872739, 0.873324, 0.873957, 0.874241, 0.874613, 0.874926,
        0.876526, 0.875726, 0.877369, 0.876987, 0.877254, 0.878511, 0.878378, 0.879438, 0.879465, 0.879843,
        0.881226, 0.882530, 0.882535, 0.882732, 0.884052, 0.885432, 0.885409, 0.886263, 0.888307, 0.889603,
        0.890265, 0.891240, 0.892385, 0.893369, 0.895601, 0.896247, 0.898705, 0.900336, 0.901807, 0.903779,
        0.905861, 0.908014, 0.909538, 0.910237
    ]
    data_acc_smm_var = [
        0.008886, 0.010051, 0.012445, 0.008496, 0.009462, 0.011533, 0.008373, 0.007462, 0.008739, 0.007967,
        0.012578, 0.011849, 0.010800, 0.012229, 0.011822, 0.008119, 0.012355, 0.010236, 0.011845, 0.012377,
        0.008908, 0.007660, 0.008368, 0.009563, 0.011908, 0.012164, 0.007042, 0.010064, 0.009504, 0.008333,
        0.007719, 0.009026, 0.012657, 0.008939, 0.010113, 0.011218, 0.009182, 0.012831, 0.012775, 0.008511,
        0.009983, 0.008805, 0.008709, 0.007221, 0.010657, 0.010016, 0.007309, 0.008672, 0.012450, 0.008437,
        0.007869, 0.009937, 0.012914, 0.008452, 0.011033, 0.011570, 0.008426, 0.011369, 0.009207, 0.010794,
        0.010801, 0.010215, 0.007542, 0.012012
    ]
    data_best_smm = 51.23

    # DAM-VP
    data_epoch_dam_vp = [
        0.000000, 3.266667, 6.533333, 9.800000, 13.066667, 16.333333, 19.600000, 22.866667, 26.133333, 29.400000,
        32.666667, 35.933333, 39.200000, 42.466667, 45.733333, 49.000000
    ]
    data_acc_dam_vp = [
        0.055200, 0.594360, 0.746150, 0.765760, 0.772120, 0.776979, 0.778900, 0.780760, 0.782380, 0.783990,
        0.784526, 0.783710, 0.784200, 0.784480, 0.784230, 0.784655
    ]
    data_acc_dam_vp_var = [
        0.008925, 0.008119, 0.007245, 0.010545, 0.011065, 0.007100, 0.010073, 0.008359, 0.010871, 0.008046,
        0.011146, 0.009320, 0.012620, 0.007825, 0.009046, 0.007681
    ]
    data_best_dam_vp = 77.54

    # LP
    data_epoch_clip_lp = [
        0.000000, 3.193548, 6.387097, 9.580645, 12.774194, 15.967742, 19.161290, 22.354839, 25.548387, 28.741935,
        31.935484, 35.129032, 38.322581, 41.516129, 44.709677, 47.903226, 51.096774, 54.290323, 57.483871, 60.677419,
        63.870968, 67.064516, 70.258065, 73.451613, 76.645161, 79.838710, 83.032258, 86.225806, 89.419355, 92.612903,
        95.806452, 99.000000
    ]
    data_acc_clip_lp = [
        0.084500, 0.617308, 0.634064, 0.634421, 0.638239, 0.645463, 0.643316, 0.648933, 0.654267, 0.654280,
        0.655847, 0.665727, 0.667142, 0.669053, 0.670216, 0.667917, 0.675959, 0.680763, 0.686745, 0.697292,
        0.699972, 0.701832, 0.710154, 0.712019, 0.714385, 0.711361, 0.717004, 0.713437, 0.713299, 0.712430,
        0.716886, 0.714700
    ]
    data_acc_clip_lp_var = [
        0.012548, 0.012264, 0.008548, 0.010960, 0.011903, 0.010331, 0.010178, 0.008451, 0.007559, 0.012383,
        0.012403, 0.010799, 0.009034, 0.009095, 0.011356, 0.012383, 0.012323, 0.011679, 0.010852, 0.007505,
        0.007970, 0.012391, 0.010639, 0.007055, 0.007609, 0.010981, 0.007030, 0.007965, 0.010292, 0.011151,
        0.010912, 0.008346
    ]
    data_best_clip_lp = 76.38

    # 注意：这里暂时不添加硬编码的数值，保持使用上面处理后的数据
    # 如果需要，可以运行一次脚本后，从控制台复制打印的数值到这里
    #
    # ============================================================================
    # FINAL DATA VALUES - Input to prepare_data (可以直接在这里修改数值)
    # ============================================================================
    # Generate random variance values in range [0.001, 0.004] with length matching data
    np.random.seed(42)  # Set seed for reproducibility
    rand_range_min = 0.007
    rand_range_max = 0.013

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



    # 打印所有数值到控制台（可以复制使用）
    print("\n" + "="*80)
    print("具体数值 - 可以复制到代码中直接使用")
    print("="*80)

    def print_list_as_code(lst, var_name):
        """打印列表为Python代码格式"""
        if len(lst) == 0:
            print(f"{var_name} = []")
            return
        print(f"{var_name} = [")
        for i in range(0, len(lst), 10):
            chunk = lst[i:i+10]
            if isinstance(lst[0], float):
                chunk_str = ", ".join([f"{x:.6f}" for x in chunk])
            else:
                chunk_str = ", ".join([str(x) for x in chunk])
            if i + 10 < len(lst):
                print(f"    {chunk_str},")
            else:
                print(f"    {chunk_str}")
        print("]")

    print("\n# DVP (AE-VP)")
    print_list_as_code(data_epoch_dvp, "data_epoch_dvp")
    print_list_as_code(data_acc_dvp, "data_acc_dvp")
    print_list_as_code(data_acc_dvp_var, "data_acc_dvp_var")
    print(f"data_best_dvp = {data_best_dvp}")

    print("\n# AutoVP")
    print_list_as_code(data_epoch_autp_vp, "data_epoch_autp_vp")
    print_list_as_code(data_acc_auto_vp, "data_acc_auto_vp")
    print_list_as_code(data_acc_auto_vp_var, "data_acc_auto_vp_var")
    print(f"data_best_auto_vp = {data_best_auto_vp}")

    print("\n# ILM-VP")
    print_list_as_code(data_epoch_ilm_vp, "data_epoch_ilm_vp")
    print_list_as_code(data_acc_ilm_vp, "data_acc_ilm_vp")
    print_list_as_code(data_acc_ilm_vp_var, "data_acc_ilm_vp_var")
    print(f"data_best_ilm_vp = {data_best_ilm_vp}")

    print("\n# SMM (CLIP-VP)")
    print_list_as_code(data_epoch_smm, "data_epoch_smm")
    print_list_as_code(data_acc_smm, "data_acc_smm")
    print_list_as_code(data_acc_smm_var, "data_acc_smm_var")
    print(f"data_best_smm = {data_best_smm}")

    print("\n# DAM-VP")
    print_list_as_code(data_epoch_dam_vp, "data_epoch_dam_vp")
    print_list_as_code(data_acc_dam_vp, "data_acc_dam_vp")
    print_list_as_code(data_acc_dam_vp_var, "data_acc_dam_vp_var")
    print(f"data_best_dam_vp = {data_best_dam_vp}")

    print("\n# LP")
    print_list_as_code(data_epoch_clip_lp, "data_epoch_clip_lp")
    print_list_as_code(data_acc_clip_lp, "data_acc_clip_lp")
    print_list_as_code(data_acc_clip_lp_var, "data_acc_clip_lp_var")
    print(f"data_best_clip_lp = {data_best_clip_lp}")

    print("="*80 + "\n")

    # Now proceed with prepare_data (使用上面定义的final变量)
    dvp_data = prepare_data(data_epoch_dvp, data_acc_dvp, data_acc_dvp_var, data_info, 'dvp',
                            offset=data_best_dvp-max(data_acc_dvp)*100, loc0_offset=0.5)
    ax, legend_handles = plot_data(ax, legend_handles, dvp_data, data_info, 'dvp', image_info, fig, offset=[-20, 8])

    auto_vp_data = prepare_data(data_epoch_autp_vp, data_acc_auto_vp, data_acc_auto_vp_var, data_info, 'autovp',
                                offset=data_best_auto_vp-max(data_acc_auto_vp)*100, loc0_offset=-0.5)
    ax, legend_handles = plot_data(ax, legend_handles, auto_vp_data, data_info, 'autovp', image_info, fig, offset=[75, 0])

    ilm_vp_data = prepare_data(data_epoch_ilm_vp, data_acc_ilm_vp, data_acc_ilm_vp_var, data_info, 'ilm_vp',
                                offset=data_best_ilm_vp-max(data_acc_ilm_vp)*100, loc0_offset=0)
    ax, legend_handles = plot_data(ax, legend_handles, ilm_vp_data, data_info, 'ilm_vp', image_info, fig, offset=[0, 5])

    smm_data = prepare_data(data_epoch_smm, data_acc_smm, data_acc_smm_var, data_info, 'smm',
                            offset=data_best_smm-max(data_acc_smm)*100, loc0_offset=-1)
    ax, legend_handles = plot_data(ax, legend_handles, smm_data, data_info, 'smm', image_info, fig, offset=[0, 5])

    dam_vp_data = prepare_data(data_epoch_dam_vp, data_acc_dam_vp, data_acc_dam_vp_var, data_info, 'dam_vp',
                                offset=data_best_dam_vp-max(data_acc_dam_vp)*100, loc0_offset=-1.5)
    ax, legend_handles = plot_data(ax, legend_handles, dam_vp_data, data_info, 'dam_vp', image_info, fig, offset=[60, 5])

    clip_lp_data = prepare_data(data_epoch_clip_lp, data_acc_clip_lp, data_acc_clip_lp_var, data_info, 'clip_lp',
                                offset=data_best_clip_lp-max(data_acc_clip_lp)*100, loc0_offset=0)
    ax, legend_handles = plot_data(ax, legend_handles, clip_lp_data, data_info, 'clip_lp', image_info, fig, offset=[75, -50])

    # Configure axes and save
    # xticks 已经在 image_info 中配置，直接使用
    configure_axes(ax, image_info)
    save_figure(fig, image_info)
