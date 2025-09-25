# DDIM ICLR 1x4 Analysis 更新总结

## 🎯 更新目标
将原来的1×3图例改为1×4图例，并添加score向量夹角分析。

## 📊 新的1×4布局

### 图1: XT Space PCA2 Analysis
- **内容**: x_T的PCA2轨迹
- **描述**: 图像状态在PCA空间中的演化轨迹
- **颜色**: 从绿色(开始)到橙色(结束)

### 图2: PC2/PC1 Ratio per Step  
- **内容**: x_T的PC2/PC1比例ρ
- **描述**: 每个时间步的PC2/PC1比值，显示与理论值1.0的偏差
- **特点**: 
  - 最多显示40个点（均匀采样）
  - x轴显示原始时间步长刻度
  - 红色虚线标记理论值1.0

### 图3: Score Space PCA2 Analysis
- **内容**: score的PCA2轨迹  
- **描述**: 梯度向量在PCA空间中的演化轨迹
- **颜色**: 从绿色(开始)到橙色(结束)

### 图4: Score Vector Angle Analysis
- **内容**: score_i与score_{i-1}的夹角
- **描述**: 连续score向量之间的夹角变化
- **特点**:
  - 红色虚线标记90°(π/2)
  - 显示角度变化趋势
  - 分析梯度方向的稳定性

## 🔧 技术实现

### 新增功能
1. **`calculate_score_angles()`方法**
   ```python
   def calculate_score_angles(self, trajectories):
       """计算连续score向量之间的夹角"""
       # 计算余弦相似度
       cos_angle = dot_product / (norm_curr * norm_prev)
       # 转换为角度
       angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
   ```

2. **1×4布局调整**
   - 图形尺寸: `figsize=(22, 5)` (从17×5调整)
   - 子图数量: 4个 (从3个增加)
   - 标题更新: 包含Score Angle Analysis

3. **x轴刻度修复**
   - 确保图2和图4显示原始时间步长
   - 采样点标签对应实际步数

### 代码结构
```python
class DDIMICLRAnalysis:
    def calculate_score_angles(self, trajectories):
        # 新增: score向量夹角计算
    
    def plot_iclr_1x4_analysis(self, trajectories, xt_pca, score_pca, step_ratios, score_angles):
        # 修改: 1x4布局绘图
    
    def generate_analysis_report(self, xt_pca, score_pca, step_ratios, score_angles):
        # 修改: 包含score角度分析报告
```

## 📈 分析指标

### Score角度分析
- **平均角度**: 接近90°表示梯度方向变化适中
- **角度标准差**: 衡量梯度方向稳定性
- **与90°的偏差**: 评估梯度方向的一致性

### 报告内容
```
SCORE VECTOR ANGLE ANALYSIS:
------------------------------
Mean angle: X.XXXXXX radians (XX.XX°)
Std angle: X.XXXXXX radians (XX.XX°)
Min angle: X.XXXXXX radians (XX.XX°)
Max angle: X.XXXXXX radians (XX.XX°)
Deviation from 90°: X.XXXXXX radians (XX.XX°)
```

## ✅ 测试结果
- **Score角度计算**: ✅ 通过
- **绘图功能**: ✅ 通过
- **1×4布局**: ✅ 正确显示
- **x轴刻度**: ✅ 显示原始时间步长

## 🚀 使用方法
```python
# 运行完整的1x4分析
python3 ddim_iclr_1x4_analysis.py

# 或者导入使用
from ddim_iclr_1x4_analysis import DDIMICLRAnalysis
analyzer = DDIMICLRAnalysis(n_samples=10000, num_inference_steps=500, num_trajectories=20)
```

## 📁 文件结构
- `ddim_iclr_1x4_analysis.py`: 主要的1×4分析脚本
- `test_1x4_analysis.py`: 功能测试脚本
- `DDIM_1X4_ANALYSIS_UPDATE.md`: 本更新总结文档

## 🎉 总结
成功将DDIM ICLR分析从1×3布局升级为1×4布局，新增了score向量夹角分析功能，提供了更全面的DDIM方法分析视角。所有功能经过测试验证，可以正常使用。
