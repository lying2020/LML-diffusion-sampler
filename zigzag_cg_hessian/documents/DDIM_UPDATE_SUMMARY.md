# DDIM ICLR 1x3 Analysis 更新总结

## 更新内容

### 1. PC2/PC1 比例图优化
- **问题**: 当步数很多时（如1000步），图表过于密集
- **解决方案**: 最多显示40个点，均匀采样
- **实现**: 
  ```python
  if len(steps) > 40:
      sample_indices = np.linspace(0, len(steps)-1, 40, dtype=int)
      sampled_steps = steps[sample_indices]
      sampled_ratios = step_ratios[sample_indices]
  ```

### 2. 局部轨迹窗口分析
- **新增功能**: 自动识别高斜率变化点
- **窗口大小**: 7点窗口（t-3, t-2, t-1, t, t+1, t+2, t+3）
- **选择策略**: 基于PC2/PC1比例的一阶导数（斜率）选择3个高变化点

### 3. 新增方法

#### `find_high_slope_points(step_ratios, num_points=3)`
- 计算PC2/PC1比例的一阶导数
- 找到斜率变化最大的3个点
- 确保窗口边界安全（避免越界）

#### `plot_local_window(Z2d, t, ax, title)`
- 绘制7点局部轨迹窗口
- 中心点用方块标注
- 自动调整坐标轴比例

#### `plot_local_trajectory_windows(trajectories, xt_pca, step_ratios, save_dir)`
- 生成独立的局部轨迹窗口图
- 1x3布局显示3个高斜率点
- 包含斜率信息标注

### 4. 输出文件更新

#### 新增文件
- `ddim_local_windows_*.png`: 局部轨迹窗口图
- 分析报告中增加高斜率点信息

#### 更新内容
- PC2/PC1比例图最多显示40个点
- 自动生成局部轨迹分析
- 详细的高斜率点统计信息

## 技术细节

### 高斜率点选择算法
```python
def find_high_slope_points(self, step_ratios, num_points=3):
    # 计算斜率（一阶导数）
    slopes = np.diff(step_ratios)
    
    # 找到高绝对斜率的点
    slope_indices = np.argsort(np.abs(slopes))[-num_points:]
    
    # 确保窗口边界安全
    valid_indices = []
    for idx in slope_indices:
        if 3 <= idx <= len(step_ratios) - 4:
            valid_indices.append(idx)
    
    return valid_indices[:num_points]
```

### 局部窗口可视化
- **窗口大小**: 7个连续时间步
- **中心标注**: 用方块标记中心时间步
- **斜率信息**: 在标题中显示斜率值
- **坐标轴**: 等比例显示，便于观察轨迹形状

## 使用示例

### 运行更新后的脚本
```bash
python3 ddim_iclr_1x3_analysis.py
```

### 输出文件
1. `ddim_iclr_1x3_analysis_*.png` - 主要分析图（PC2/PC1图最多40个点）
2. `ddim_local_windows_*.png` - 局部轨迹窗口图
3. `ddim_iclr_1x3_analysis_report_*.txt` - 详细分析报告

### 分析结果示例
```
High Slope Points for Local Windows:
  Point 1: t=21, slope=-0.010857
  Point 2: t=22, slope=-0.051667  
  Point 3: t=23, slope=-0.062526
```

## 优势

1. **可视化优化**: PC2/PC1比例图更清晰，避免过度密集
2. **局部分析**: 自动识别关键变化点，提供局部轨迹细节
3. **智能选择**: 基于数学斜率选择最有意义的分析点
4. **完整分析**: 结合全局和局部分析，提供全面的轨迹理解

## 兼容性

- 保持原有API不变
- 向后兼容所有现有功能
- 新增功能为可选，不影响原有分析流程
