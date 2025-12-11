# Zigzag Analysis for CIFAR-10 Diffusion Models

## 概述
这个文件夹包含了用于分析CIFAR-10扩散模型中zigzag现象的统一分析工具。

## 文件结构
```
zigzag_cg_hessian/
├── zigzag_analysis_unified.py    # 主要的分析脚本
├── backup_old_files/             # 备份的旧文件
└── README.md                     # 本文件
```

## 功能特性

### 1. PCA2降维算法
- 使用前2个PCA主成分进行分析
- 基于score向量的协方差矩阵进行PCA
- 选择最大和最小特征值对应的方向以最大化zigzag可见性

### 2. 采样配置
- **采样步数**: 20步
- **样本数量**: 100个
- **模型**: CIFAR-10 DDPM预训练模型
- **支持方法**: ["pndm", "ddim", "dpm", "unipc"]

### 3. 生成图表
- **轨迹图**: PCA空间中的zigzag折线图
- **路径图**: 轨迹在2D空间的投影
- **收敛分析图**: 到终点的距离变化
- **特征值谱图**: PCA特征值分布
- **比较图**: 所有方法的综合比较

### 4. 评价指标
- **平均夹角** (Mean Direction Angle): 相邻步骤间的平均角度
- **离散曲率** (Discrete Curvature): 单位方向向量的变化
- **Zigzag得分** (Zigzag Score): 角度和曲率的综合指标
- **各向异性程度** (Anisotropy Ratio): λ₂/λ₁比值
- **最终距离** (Final Distance): 到原点的平均距离

## 使用方法

### 基本使用
```bash
cd /home/liying/Desktop/LML-diffusion-sampler/zigzag_cg_hessian
python3 zigzag_analysis_unified.py
```

### 自定义参数
```python
# 在脚本中修改参数
analyzer = ZigzagAnalyzer(
    n_samples=100,           # 样本数量
    num_inference_steps=20,  # 推理步数
    device='cuda'            # 设备
)
```

## 输出文件

### 图片文件
- `zigzag_trajectories_{method}_{timestamp}.png`: 各方法的轨迹图
- `convergence_analysis_{method}_{timestamp}.png`: 各方法的收敛分析图
- `zigzag_comparison_{timestamp}.png`: 所有方法的比较图

### 数据文件
- `zigzag_analysis_results_{timestamp}.json`: 详细的数值结果
- `zigzag_analysis_report_{timestamp}.txt`: 文本格式的分析报告

### 日志文件
- 保存在 `output/logs/` 目录下

## 理论基础

### PCA2降维原理
1. **数据收集**: 在特定时间t收集一批样本的score向量
2. **协方差计算**: 计算score向量的协方差矩阵
3. **特征分解**: 对协方差矩阵进行特征值分解
4. **方向选择**: 选择最大和最小特征值对应的特征向量
5. **投影**: 将所有轨迹投影到这两个方向构成的2D空间

### Zigzag评价指标
1. **方向夹角**: θₖ = arccos(⟨Δxₖ, Δxₖ₋₁⟩ / (||Δxₖ|| ||Δxₖ₋₁||))
2. **离散曲率**: κₖ = ||uₖ - uₖ₋₁||，其中 uₖ = Δxₖ / ||Δxₖ||
3. **各向异性**: ρ = λ₂/λ₁，比值越大表示各向异性越强

## 注意事项

1. **内存使用**: 100个样本的轨迹分析需要较多内存，建议使用GPU
2. **模型路径**: 确保CIFAR-10 DDPM模型路径正确
3. **可视化**: 生成的图片采用ICLR论文格式，适合学术发表
4. **数值稳定性**: 代码中包含了数值稳定性检查，避免除零错误

## 与旧版本的区别

### 改进点
1. **统一接口**: 所有分析功能集成在一个脚本中
2. **标准化输出**: 采用ICLR论文格式的图表
3. **完整指标**: 包含所有重要的zigzag评价指标
4. **错误处理**: 完善的异常处理和日志记录
5. **模块化设计**: 易于扩展和维护

### 删除的冗余文件
- 所有旧的PNG图片文件
- 重复的分析脚本
- 过时的报告文件
- 临时文本文件

## 扩展功能

### 添加新方法
```python
# 在sampler_types中添加新方法
self.sampler_types = ["pndm", "ddim", "dpm", "unipc", "new_method"]
```

### 自定义评价指标
```python
def compute_custom_metrics(self, projected_trajectories):
    # 添加自定义的zigzag评价指标
    pass
```

### 修改可视化
```python
def plot_custom_visualization(self, data):
    # 添加自定义的可视化图表
    pass
```
