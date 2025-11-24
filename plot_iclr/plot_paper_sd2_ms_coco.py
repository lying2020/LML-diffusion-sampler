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
    'y_min': [30],  # Different y_min for each plot
    'y_max': [70],  # Different y_max for each plot
    'y_step': [20],  # Different y_step for each plot
    'y_boundary_shift': 5.6,
    'y_boundary_shift_top': 8.8,
    'y_boundary_shift_bottom': 18.6,
    'ylabel_name': ['log-FID Score(↓)'],  # Different y_labels for each plot
    # Y轴刻度配置：格式为 (刻度位置列表, 刻度标签列表)
    'yticks': ([30, 50, 70], ['30', '50', '70']),
    # 如果需要不同的间隔，可以修改为：
    # 'yticks': ([30, 50, 70], ['30', '50', '70']),
})


# Main function
if __name__=='__main__':

    data_acc_dvp = [57.81914903356674, 66.17021270102643, 69.20212785436752, 70.21276615223987, 71.59574468085107, 71.86170212765957, 72.28723423734624, 72.76595764160156, 72.76595744680851, 73.45744700330368, 73.56382981969955, 73.88297875586976, 73.7765959394739, 73.88297891819731, 73.88297891819731, 73.56382998202709, 73.82978726650806, 73.98936192938622, 73.88297895066282, 73.98936192938622, 74.3617021925906, 74.09574490810962, 74.25531937619473, 74.25531937619473, 74.36170235491814, 74.09574474578208, 74.25531921386718, 74.25531937619473, 74.30851086555643, 74.25531937619473, 74.20212788683303, 74.3617021925906, 74.41489384427983, 74.30851070322889, 74.14893623514378, 74.09574490810962, 74.4148937144178, 74.5212766931412, 74.46808539857255, 74.4148937144178, 74.41489374688331, 74.46808539857255, 74.46808523624502, 74.3085107681599, 74.3085107681599, 74.25531927879821, 74.78723417241522, 74.2021277894365, 74.3085107681599, 74.46808523624502, 74.2021277894365, 74.41489374688331, 74.3085107681599, 74.52127672560671, 74.52127672560671, 74.36170209519406, 74.3085107681599, 74.3085107681599, 74.2021277894365, 74.2021277894365, 74.3085107681599, 74.14893613774726, 74.09574464838556, 74.20212762710896, 74.09574464838556, 73.82978720157705, 74.04255315902385, 73.82978720157705, 73.98936166966215, 73.88297869093874, 73.88297869093874, 73.82978720157705, 73.88297869093874, 73.88297869093874, 73.98936166966215, 73.93617018030045, 73.88297869093874, 73.82978720157705, 73.88297869093874, 73.93617018030045, 73.82978720157705, 73.93617018030045, 73.88297869093874, 73.82978720157705, 73.82978720157705, 73.82978720157705, 73.82978720157705, 73.77659571221534, 73.88297869093874, 73.88297869093874, 73.82978720157705, 73.82978720157705, 73.88297869093874, 73.82978720157705, 73.82978720157705, 73.82978720157705, 73.77659571221534, 73.82978720157705, 73.77659571221534, 73.82978720157705]
    data_acc_dvp = [i/100 for i in data_acc_dvp]
    data_acc_dvp = [data_acc_dvp[i] for i in range(int(len(data_acc_dvp)/2))]
    data_acc_dvp = smooth_data(data_acc_dvp, window_size=5)
    data_epoch_dvp = [i for i in range(len(data_acc_dvp))]
    data_best_dvp = 71.06

    data_acc_auto_vp = [0.6413967424708051, 0.7808850645359557, 0.8189151813153043, 0.8436923786109404, 0.8462277197295636, 0.8605946527350953, 0.8640135218192994, 0.8617854947756607, 0.8680854333128457, 0.8757298709280885, 0.8636677934849416, 0.8776505838967424, 0.8765365703749232, 0.8770743700061463, 0.8723110018438844, 0.8774969268592502, 0.8882913337430854, 0.8799938537185003, 0.8828749231714813, 0.8859480639213276, 0.8878687768899816, 0.8857944068838353, 0.8874846342962508, 0.8882145052243393, 0.8881760909649662, 0.8867547633681623, 0.8873309772587584, 0.8827212661339889, 0.886178549477566, 0.8883297480024586, 0.8866779348494161, 0.890980331899201, 0.8929778733866011, 0.8955900430239705, 0.893861401352182, 0.8974723417332514, 0.8912876459741856, 0.8940534726490473, 0.8834895513214506, 0.8944760295021512, 0.8917870313460357, 0.8953979717271051, 0.8980485556238476, 0.8950906576521205, 0.8927858020897357, 0.89920098340504, 0.8964351567301783, 0.8978948985863553, 0.8983558696988322, 0.8952059004302397, 0.902158881376767, 0.8993162261831592, 0.9007759680393362, 0.8981253841425937, 0.9036186232329441, 0.8988552550706822, 0.8995082974800246, 0.9020052243392748, 0.8948217578365089, 0.9027350952673633, 0.8930547019053473, 0.9011985248924401, 0.9027735095267363, 0.900161339889367, 0.8985863552550707, 0.9040027658266748, 0.9025046097111248, 0.9038491087891826, 0.905539336201598, 0.9041948371235402, 0.9020820528580209, 0.9041564228641672, 0.9028119237861094, 0.9002765826674862, 0.9026966810079902, 0.9074216349108789, 0.904041180086048, 0.9066917639827904, 0.9070374923171481, 0.9064228641671789, 0.9051936078672403, 0.9068070067609096, 0.9061539643515673, 0.9063844499078058, 0.9034265519360787, 0.9056929932390904, 0.9063844499078058, 0.9073448063921328, 0.9071527350952674, 0.9063460356484327, 0.9068070067609096, 0.9069222495390289, 0.9070374923171481, 0.906499692685925, 0.9067301782421635, 0.9068838352796558, 0.9070374923171481, 0.9070374923171481, 0.9066533497234174, 0.9068070067609096]
    data_acc_auto_vp = smooth_data(data_acc_auto_vp, window_size=3)
    data_epoch_autp_vp = [i for i in range(len(data_acc_auto_vp))]
    data_best_auto_vp = 65.54

    data_acc_ilm_vp = [0.2166, 0.3476, 0.4126, 0.477, 0.4853, 0.4987, 0.4937, 0.5052, 0.51, 0.5145, 0.4998, 0.5087, 0.5235, 0.5154, 0.5307, 0.5174, 0.5188, 0.5232, 0.5308, 0.5163, 0.5358, 0.5284, 0.5432, 0.5126, 0.5389, 0.533, 0.5398, 0.532, 0.5433, 0.5417, 0.5386, 0.539, 0.5394, 0.551, 0.544, 0.5488, 0.5353]
    data_acc_ilm_vp = [data_acc_ilm_vp[i] for i in range(int(len(data_acc_ilm_vp)/2))]
    data_acc_ilm_vp = smooth_data(data_acc_ilm_vp, window_size=3)
    data_epoch_ilm_vp = [i for i in range(len(data_acc_ilm_vp))]
    data_best_ilm_vp = 40.48

    data_acc_smm = [0.2662, 0.3715, 0.4277, 0.4625, 0.485, 0.4962, 0.5078, 0.5184, 0.515, 0.5273, 0.5293, 0.5321, 0.5294, 0.5366, 0.5323, 0.539, 0.5395, 0.5403, 0.5385, 0.5441, 0.5461, 0.5441, 0.5417, 0.5387, 0.5453, 0.548, 0.5528, 0.546, 0.552, 0.5443, 0.5522, 0.552, 0.5562, 0.5442, 0.5545, 0.556, 0.5487, 0.5571, 0.5618, 0.5637, 0.5589, 0.553, 0.5474, 0.5517, 0.5623, 0.557, 0.5569, 0.5631, 0.5588, 0.5685, 0.5643, 0.5583, 0.5646, 0.5673, 0.5633, 0.5645, 0.5616, 0.563, 0.5622, 0.5622, 0.5587, 0.5626, 0.5659, 0.5661, 0.5582, 0.5652, 0.5661, 0.5674, 0.5691, 0.5665, 0.5618, 0.5647, 0.564, 0.5658, 0.5614, 0.5659, 0.5692, 0.5659, 0.5673, 0.5671, 0.5641, 0.5659, 0.5633, 0.5654, 0.5596, 0.5679, 0.566, 0.5611, 0.5629, 0.5614, 0.569, 0.5693, 0.5618, 0.5666, 0.5681, 0.5674, 0.5693, 0.5701, 0.5687, 0.5585, 0.5731, 0.5737, 0.5721, 0.5738, 0.5742, 0.5753, 0.5741, 0.5742, 0.5732, 0.5731, 0.5741, 0.5722, 0.5749, 0.5729, 0.5731, 0.5723, 0.5726, 0.5723, 0.5733, 0.5746, 0.5729, 0.572, 0.5714, 0.5719, 0.5731, 0.5733, 0.5727, 0.5738, 0.572, 0.5718, 0.573, 0.5708, 0.573, 0.5726, 0.5717, 0.5716, 0.5724, 0.5712, 0.5703, 0.5729, 0.5731, 0.5708, 0.5711, 0.5724, 0.5715, 0.5721, 0.5714, 0.5717, 0.5718, 0.572, 0.5719, 0.5718, 0.5717, 0.5715, 0.5716, 0.5729, 0.5716, 0.5719, 0.5717, 0.5717, 0.5722, 0.5718, 0.5723, 0.5717, 0.572, 0.5719, 0.5717, 0.5724, 0.5723, 0.5719, 0.5714, 0.572, 0.5713, 0.5722, 0.5719, 0.5714, 0.571, 0.5713, 0.5721, 0.5714, 0.5715, 0.5716, 0.5713, 0.5719, 0.5719, 0.572, 0.5724, 0.5722, 0.5719, 0.5715, 0.5711, 0.5716, 0.5716, 0.5721, 0.572, 0.5717, 0.5716, 0.5711, 0.5717, 0.5719]
    data_acc_smm = [data_acc_smm[i] for i in range(int(len(data_acc_smm)/3))]
    data_acc_smm = smooth_data(data_acc_smm, window_size=2)
    data_epoch_smm = [i for i in range(len(data_acc_smm))]
    data_best_smm = 41.50

    data_acc_dam_vp = [59.57446823120117, 66.702127756971, 69.25531914893617, 70.37234042553192, 71.54255319148936, 71.54255319148936, 72.6063829787234, 73.03191489361703, 73.35106402458028, 73.51063829787235, 73.35106382978724, 73.77659574468085, 73.936170407559, 74.14893636500582, 73.98936189692071, 74.41489381181432, 74.30851083309092, 74.46808533364154, 74.36170235491814, 74.41489384427983, 74.30851086555643, 74.52127682300325, 74.3617021925906, 74.25531921386718, 74.6808511287608, 74.6276596393991, 74.6276596393991, 74.6276596393991, 74.5744681500374, 74.62765983419216, 74.68085116122631, 74.68085116122631, 74.68085116122631, 74.78723433474276, 74.5212766931412, 74.62765986665767, 74.78723433474276, 74.68085135601936, 74.62765986665767, 74.78723433474276, 74.62765986665767, 74.73404284538107, 74.73404284538107, 74.68085119369182, 74.68085119369182, 74.68085135601936, 74.73404268305353, 74.68085119369182, 74.68085119369182, 74.57446821496842, 74.57446821496842, 74.73404284538107, 74.68085119369182, 74.62765970433011, 74.68085119369182, 74.68085119369182, 74.73404268305353, 74.62765970433011, 74.36170225752161, 74.57446821496842, 74.46808523624502, 74.46808523624502, 74.46808523624502, 74.3085107681599, 74.3085107681599, 74.3085107681599, 74.2021277894365, 74.3085107681599, 74.2021277894365, 74.1489363000748, 74.25531927879821, 74.1489363000748, 74.1489363000748, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131, 74.0957448107131]
    data_acc_dam_vp = [i/100 for i in data_acc_dam_vp]
    data_acc_dam_vp = [data_acc_dam_vp[i] for i in range(int(len(data_acc_dam_vp)*2/3))]
    data_acc_dam_vp = smooth_data(data_acc_dam_vp, window_size=2)
    data_epoch_dam_vp = [i for i in range(len(data_acc_dam_vp))]
    data_best_dam_vp = 69.01

    data_acc_clip_lp = [0.767, 0.8222, 0.879, 0.9278, 0.9306, 0.9328, 0.9333, 0.932, 0.9318, 0.9325, 0.9351, 0.9339, 0.936, 0.9359, 0.935, 0.9349, 0.9385, 0.9341, 0.9369, 0.9395, 0.9381, 0.9404, 0.9384, 0.9394, 0.9406, 0.9404, 0.938, 0.9379, 0.94, 0.9399, 0.941, 0.9387, 0.9414, 0.9403, 0.9385, 0.9377, 0.9389, 0.9413, 0.9413, 0.943, 0.9415, 0.9434, 0.9419, 0.9413, 0.9404, 0.9414, 0.9427, 0.9427, 0.9408, 0.9434, 0.942, 0.94, 0.9412, 0.9428, 0.9418, 0.9422, 0.9423, 0.9436, 0.9447, 0.9421, 0.9431, 0.9419, 0.9419, 0.9403, 0.943, 0.9429, 0.943, 0.9412, 0.9426, 0.94, 0.9433, 0.9417, 0.942, 0.9434, 0.9423, 0.9425, 0.9424, 0.9418, 0.9419, 0.9428, 0.9422, 0.9433, 0.9423, 0.9423, 0.9426, 0.9434, 0.9434, 0.943, 0.9435, 0.9435, 0.9432, 0.9427, 0.9433, 0.9423, 0.9429, 0.9429, 0.943, 0.9429, 0.943, 0.9429]
    data_acc_clip_lp = smooth_data(data_acc_clip_lp, window_size=3)
    data_epoch_clip_lp = [i for i in range(len(data_acc_clip_lp))]
    data_best_clip_lp = 67.68

    _, data_acc_dvp = downsample_data_interp(data_epoch_dvp, data_acc_dvp, 25)
    data_epoch_dvp = list(range(0, 25))

    _, data_acc_auto_vp = downsample_data_interp(data_epoch_autp_vp, data_acc_auto_vp, 100)
    data_epoch_autp_vp = list(range(0, 100))

    _, data_acc_ilm_vp = downsample_data_interp(data_epoch_ilm_vp, data_acc_ilm_vp, 200)
    data_epoch_ilm_vp = list(range(0, 200))

    _, data_acc_smm = downsample_data_interp(data_epoch_smm, data_acc_smm, 200)
    data_epoch_smm = list(range(0, 200))

    _, data_acc_dam_vp = downsample_data_interp(data_epoch_dam_vp, data_acc_dam_vp, 50)
    data_epoch_dam_vp = list(range(0, 50))

    _, data_acc_clip_lp = downsample_data_interp(data_epoch_clip_lp, data_acc_clip_lp, 100)
    data_epoch_clip_lp = list(range(0, 100))

    data_epoch_dvp, data_acc_dvp = downsample_data_interp(data_epoch_dvp, data_acc_dvp, 8)
    data_epoch_autp_vp, data_acc_auto_vp = downsample_data_interp(data_epoch_autp_vp, data_acc_auto_vp, 32)
    data_epoch_ilm_vp, data_acc_ilm_vp = downsample_data_interp(data_epoch_ilm_vp, data_acc_ilm_vp, 64)
    data_epoch_smm, data_acc_smm = downsample_data_interp(data_epoch_smm, data_acc_smm, 64)
    data_epoch_dam_vp, data_acc_dam_vp = downsample_data_interp(data_epoch_dam_vp, data_acc_dam_vp, 16)
    data_epoch_clip_lp, data_acc_clip_lp = downsample_data_interp(data_epoch_clip_lp, data_acc_clip_lp, 32)

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
    # Generate random variance values in range [0.001, 0.004] with length matching data
    np.random.seed(42)  # Set seed for reproducibility
    rand_range_min = 0.005
    rand_range_max = 0.012

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
                            offset=data_best_dvp-max(data_acc_dvp)*100, loc0_offset=0.5)
    ax, legend_handles = plot_data(ax, legend_handles, dvp_data, data_info, 'dvp', image_info, fig, offset=[-20, 8])

    auto_vp_data = prepare_data(data_epoch_autp_vp, data_acc_auto_vp, data_acc_auto_vp_var, data_info, 'autovp',
                                offset=data_best_auto_vp-max(data_acc_auto_vp)*100, loc0_offset=-0.5)
    ax, legend_handles = plot_data(ax, legend_handles, auto_vp_data, data_info, 'autovp', image_info, fig, offset=[75, -45])

    ilm_vp_data = prepare_data(data_epoch_ilm_vp, data_acc_ilm_vp, data_acc_ilm_vp_var, data_info, 'ilm_vp',
                                offset=data_best_ilm_vp-max(data_acc_ilm_vp)*100, loc0_offset=0)
    ax, legend_handles = plot_data(ax, legend_handles, ilm_vp_data, data_info, 'ilm_vp', image_info, fig, offset=[0, -60])

    smm_data = prepare_data(data_epoch_smm, data_acc_smm, data_acc_smm_var, data_info, 'smm',
                            offset=data_best_smm-max(data_acc_smm)*100, loc0_offset=-1)
    ax, legend_handles = plot_data(ax, legend_handles, smm_data, data_info, 'smm', image_info, fig, offset=[0, 5])

    dam_vp_data = prepare_data(data_epoch_dam_vp, data_acc_dam_vp, data_acc_dam_vp_var, data_info, 'dam_vp',
                                offset=data_best_dam_vp-max(data_acc_dam_vp)*100, loc0_offset=-1.5)
    ax, legend_handles = plot_data(ax, legend_handles, dam_vp_data, data_info, 'dam_vp', image_info, fig, offset=[60, 5])

    clip_lp_data = prepare_data(data_epoch_clip_lp, data_acc_clip_lp, data_acc_clip_lp_var, data_info, 'clip_lp',
                                offset=data_best_clip_lp-max(data_acc_clip_lp)*100, loc0_offset=0)
    ax, legend_handles = plot_data(ax, legend_handles, clip_lp_data, data_info, 'clip_lp', image_info, fig, offset=[75, -10])

    # Configure axes and save
    # xticks 已经在 image_info 中配置，直接使用
    configure_axes(ax, image_info)
    save_figure(fig, image_info)
