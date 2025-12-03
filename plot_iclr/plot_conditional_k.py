import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import json

import project
from plot_utils import get_image_info, setup_figure, save_figure, configure_axes




# CIFAR10: 5, 7， 10   ||  16.81, 10.67， 6.24


# lambda_values = [20, 50, 100, 200, 400, 500, 1000, 1600, 2000, 2500]

lambda_values = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
# NFE=5 (Blue line) - FID values
# Pattern: decrease then increase, minimum at position 5 (2000) = 16.81
fid_nfe10 = [31.93, 31.10, 28.0, 25.0, 22.5, 20.0, 18.0, 17.2, 16.81, 16.77]
baseline_nfe10 = 32.63  # Horizontal dashed line at first point (400)

# NFE=10 (Red line) - FID values
# Pattern: decrease then increase, minimum at position 5 (2000) = 6.24
fid_nfe20 = [16.28, 16.18, 15.3, 14.8, 14.1, 13.2, 12.2, 11.5, 10.77, 10.87]
baseline_nfe20 = 16.21  # Horizontal dashed line at first point (400)

# NFE=20 (Green line) - FID values
# Pattern: decrease then increase, minimum at position 5 (2000) = 4.02
fid_nfe50 = [11.14, 10.9, 10.3, 9.9, 9.2, 8.8, 8.1, 7.2, 6.24, 6.25]
baseline_nfe50 = 11.14  # Horizontal dashed line at first point (400)

# Convert to numpy arrays and filter out None values
lambda_array = np.array(lambda_values)
fid_nfe10_array = np.array([v for v in fid_nfe10 if v is not None])
fid_nfe20_array = np.array(fid_nfe20)
fid_nfe50_array = np.array(fid_nfe50)

# Apply log transformation to FID values
fid_nfe10_array = np.log(fid_nfe10_array)
fid_nfe20_array = np.log(fid_nfe20_array)
fid_nfe50_array = np.log(fid_nfe50_array)

# Apply log transformation to baseline values
baseline_nfe10 = np.log(baseline_nfe10)
baseline_nfe20 = np.log(baseline_nfe20)
baseline_nfe50 = np.log(baseline_nfe50)

# Lambda values corresponding to valid FID values
lambda_nfe10 = lambda_array[:len(fid_nfe10_array)]
lambda_nfe20 = lambda_array
lambda_nfe50 = lambda_array

# Print data summary
print("\n" + "="*80)
print("DATA SUMMARY - FID vs. λ in HILDA")
print("="*80)
print(f"\nLambda values: {lambda_values}")
print(f"\nNFE=10 FID values: {fid_nfe10_array}")
print(f"NFE=20 FID values: {fid_nfe20_array}")
print(f"NFE=50 FID values: {fid_nfe50_array}")
print("="*80 + "\n")


# Define data-specific image information
# 只定义与数据相关的参数，其他样式配置使用 plot_utils.py 中的默认值
image_info = get_image_info({
    # 必须提供：保存文件名
    'save_title': 'conditional_k',
    # 可选：图表标题
    'image_title': 'Conditional number $K^*$ in HILDA',
    'width': 16,
    'height': 9,
    'fontsize': 40,
    'use_seaborn': 0,
    'use_times_newroman': 0,
    'grid_color': '#e0e0e8',
    'grid_alpha': 0.4,
    'grid_linewidth': 2.1,
    'background_color': 'white',
    # X轴配置
    'x_min': 200,
    'x_max': 2000,
    'x_step': 500,
    'x_boundary_shift': 40,
    'x_boundary_shift_left': 40,
    'x_boundary_shift_right': 40,
    'xlabel_name': '$K^*$',
    # X轴刻度配置：格式为 (刻度位置列表, 刻度标签列表) - 使用更稀疏的刻度
    'xticks': ([200, 400, 800, 1200, 1600, 2000], ['20', '50', '200', '800', '1600', '2400']),
    'y_min': [1.7],  # log-FID range
    'y_max': [3.6],  # log-FID range
    'y_step': [0.2],
    'y_boundary_shift': 0.1,
    'y_boundary_shift_top': 0.05,
    'y_boundary_shift_bottom': 0.05,
    'ylabel_name': ['log-FID Score(↓)'],
    # Y轴刻度配置：使用更稀疏的刻度
    'yticks': ([2.0, 2.5, 3.0, 3.5], ['2.0', '2.5', '3.0', '3.5']),
})

# lambda_values = [20, 50, 100, 200, 400, 500, 1000, 1600, 2000, 2500]

# Create figure using plot_utils
fig, ax = setup_figure(image_info)
fig.patch.set_facecolor('white')

# Define colors for each NFE line
# Based on plot_utils.py colors but slightly modified for better harmony
# Original colors: '#2ad4af' (teal), '#27a6cc' (sky blue), '#98be2c' (lime green),
#                  '#00b168' (deep green), '#fcc5c5' (pink), '#fcbd60' (orange)
color_nfe10 = '#3bb8d0'  # Modified sky blue (based on '#27a6cc', slightly lighter and more cyan)
color_nfe20 = '#ff9f66'  # Modified orange (based on '#fcbd60', slightly more red)
color_nfe50 = '#4dd4a0'  # Modified teal green (based on '#2ad4af', slightly darker)

# Plot the three main data lines with markers (no labels since no legend)
ax.plot(lambda_nfe10, fid_nfe10_array,
        color=color_nfe10, linewidth=6, linestyle='-', marker='o', markersize=8,
        zorder=3)

ax.plot(lambda_nfe20, fid_nfe20_array,
        color=color_nfe20, linewidth=6, linestyle='-', marker='o', markersize=8,
        zorder=3)

ax.plot(lambda_nfe50, fid_nfe50_array,
        color=color_nfe50, linewidth=6, linestyle='-', marker='o', markersize=8,
        zorder=3)

# Mark the minimum points with special symbols
# Find minimum point for NFE=5 (Blue line)
min_idx_10 = np.argmin(fid_nfe10_array)
ax.scatter([lambda_nfe10[min_idx_10]], [fid_nfe10_array[min_idx_10]],
          color=color_nfe10, s=400, marker='*', edgecolors='white', linewidths=2,
          zorder=6)

# Find minimum point for NFE=7 (Red line)
min_idx_20 = np.argmin(fid_nfe20_array)
ax.scatter([lambda_nfe20[min_idx_20]], [fid_nfe20_array[min_idx_20]],
          color=color_nfe20, s=400, marker='*', edgecolors='white', linewidths=2,
          zorder=6)

# Find minimum point for NFE=10 (Green line)
min_idx_50 = np.argmin(fid_nfe50_array)
ax.scatter([lambda_nfe50[min_idx_50]], [fid_nfe50_array[min_idx_50]],
          color=color_nfe50, s=400, marker='*', edgecolors='white', linewidths=2,
          zorder=6)

# Add NFE labels next to each line (at the first data point, on the left side)
# NFE=5 (Blue line)
first_idx_10 = 0
# Add small offset to move label slightly inside (to the right of the point)
x_offset = (max(lambda_values) - min(lambda_values)) * 0.03  # 3% of x-axis range
ax.text(lambda_nfe10[first_idx_10] + x_offset, fid_nfe10_array[first_idx_10]-0.15, 'NFE=5',
        color=color_nfe10, fontsize=28, fontweight='bold',
        ha='left', va='center', zorder=5)

# NFE=7 (Red line) - based on comment "5, 7， 10"
first_idx_20 = 0
ax.text(lambda_nfe20[first_idx_20] + x_offset, fid_nfe20_array[first_idx_20]-0.15, 'NFE=7',
        color=color_nfe20, fontsize=28, fontweight='bold',
        ha='left', va='center', zorder=5)

# NFE=10 (Green line)
first_idx_50 = 0
ax.text(lambda_nfe50[first_idx_50] + x_offset, fid_nfe50_array[first_idx_50]-0.15, 'NFE=10',
        color=color_nfe50, fontsize=28, fontweight='bold',
        ha='left', va='center', zorder=5)

# Plot horizontal dashed baseline lines for each NFE (no label to exclude from legend)
x_min = min(lambda_values)
x_max = max(lambda_values)
# ax.plot([x_min, x_max], [baseline_nfe10, baseline_nfe10],
#         color=color_nfe10, linestyle='--', linewidth=2, alpha=0.7,
#         zorder=1)  # No label to avoid showing in legend

# ax.plot([x_min, x_max], [baseline_nfe20, baseline_nfe20],
#         color=color_nfe20, linestyle='--', linewidth=2, alpha=0.7,
#         zorder=1)  # No label to avoid showing in legend

# ax.plot([x_min, x_max], [baseline_nfe50, baseline_nfe50],
#         color=color_nfe50, linestyle='--', linewidth=2, alpha=0.7,
#         zorder=1)  # No label to avoid showing in legend

# Customize axes
lambda_symbol = fr'$\mathbf{{K^*}}$'
ax.set_xlabel(f'{lambda_symbol}', fontsize=28, fontweight='bold')
ax.set_ylabel('log-FID', fontsize=28, fontweight='bold')
ax.set_title(f'log-FID vs. {lambda_symbol} in HILDA',
            fontsize=32, fontweight='bold', pad=15)

# Use configure_axes to apply ticks from image_info
# This will use the xticks and yticks defined in image_info
configure_axes(ax, image_info,
               xticks=image_info.get('xticks'),
               yticks=image_info.get('yticks'))

# Customize border (spines) - override default to use lighter gray
frame_color = '#b0b0b0'  # Light gray color (lighter than default #d0d0e0)
border_linewidth = 3
for spine in ax.spines.values():
    spine.set_color(frame_color)
    spine.set_linewidth(border_linewidth)

# Grid is already set by setup_figure, but we adjust it here
ax.grid(True, alpha=0.4, linewidth=image_info['grid_linewidth'], linestyle='-', zorder=0)
ax.set_axisbelow(True)


# Customize tick parameters - double font size and make bold
# Remove all tick marks (both internal and external) - keep only labels
ax.tick_params(axis='both', which='major', labelsize=36, width=1.5, length=0,
               top=False, right=False)  # length=0 removes all tick marks
ax.tick_params(axis='both', which='minor', labelsize=28, width=1, length=0,
               top=False, right=False)

# Make tick labels bold
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# Add minor ticks for better readability
ax.minorticks_on()

# Save generated data to file for user to review and adjust
# Note: FID values are stored as log-transformed values for plotting
data_output = {
    'lambda_values': lambda_values,
    'fid_nfe10_original': fid_nfe10,  # Original FID values
    'fid_nfe20_original': fid_nfe20,  # Original FID values
    'fid_nfe50_original': fid_nfe50,  # Original FID values
    'fid_nfe10_log': fid_nfe10_array.tolist(),  # Log-transformed FID values
    'fid_nfe20_log': fid_nfe20_array.tolist(),  # Log-transformed FID values
    'fid_nfe50_log': fid_nfe50_array.tolist(),  # Log-transformed FID values
    'baseline_nfe10_original': np.exp(baseline_nfe10),  # Original baseline value
    'baseline_nfe20_original': np.exp(baseline_nfe20),  # Original baseline value
    'baseline_nfe50_original': np.exp(baseline_nfe50),  # Original baseline value
    'baseline_nfe10_log': float(baseline_nfe10),
    'baseline_nfe20_log': float(baseline_nfe20),
    'baseline_nfe50_log': float(baseline_nfe50),
}
with open('fid_lambda_data.json', 'w') as f:
    json.dump(data_output, f, indent=2)
print(f"✓ Generated data saved to: fid_lambda_data.json")
print(f"  You can edit this file to adjust the values, then reload in the script.\n")

# Save PNG with high DPI first (before save_figure closes the figure)
png_path = os.path.join(project.results_iclr_path, f'{image_info["save_title"]}.png')
fig.savefig(png_path, dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none', format='png')
print(f"✓ PNG saved: {png_path}")

# Save PDF and JPG using plot_utils.save_figure (this will close the figure)
save_figure(fig, image_info)
