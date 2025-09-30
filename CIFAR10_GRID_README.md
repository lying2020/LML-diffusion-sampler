# CIFAR-10 对比图组生成功能

## 功能概述

已成功在 `scripts/cifar10.py` 中添加了 `--generate_grid` 功能，可以生成类似截图的6行多列图像对比，展示不同采样方法生成的CIFAR-10图像效果。

## 新增功能

### 1. `--generate_grid` 命令行参数
- 从现有图像文件生成对比图组
- 自动检测可用的采样器
- 无需重新生成图像

### 2. `generate_comparison_grid()` 函数
生成对比图组的主要函数。

**参数：**
- `save_dir`: 图像保存目录
- `sampler_types`: 采样器类型列表，如 `['ddim', 'pndm', 'dpm', 'dpm++', 'unipc', 'dpm_lm']`
- `num_inference_steps`: 推理步数，默认20
- `test_num`: 行数（图像数量），默认6
- `batch_size`: 批次大小，默认1

**功能：**
- 创建6行多列的图像网格
- 每列代表一种采样方法
- 每行代表同一个主题的不同生成结果
- 自动添加方法标签和分隔线
- 支持LML方法后的虚线分隔

### 3. `generate_comparison_grid_from_existing()` 函数
从已存在的图像文件生成对比图组。

**参数：**
- `save_dir`: 图像保存目录
- `num_inference_steps`: 推理步数，默认20

**功能：**
- 自动检测可用的采样器
- 从现有图像文件生成对比图组
- 无需重新生成图像

## 使用方法

### 方法1：命令行使用

```bash
# 生成对比图组（需要先有图像文件）
python scripts/cifar10.py --generate_grid --num_inference_steps 20

# 运行批量实验后生成对比图组
python scripts/cifar10.py --run_batch --num_inference_steps 20
python scripts/cifar10.py --generate_grid --num_inference_steps 20
```

### 方法2：Python代码调用

```python
from scripts.cifar10 import generate_comparison_grid_from_existing

# 生成对比图组
output_path = generate_comparison_grid_from_existing(
    'output/cifar10', 
    num_inference_steps=20
)
```

## 输出结果

### 对比图组
- **文件位置**: `output/cifar10/comparison_grid_steps{num_inference_steps}.png`
- **布局**: 6行6列（6行图像，6种采样方法）
- **方法顺序**: DDIM, PNDM, DPM-Solver, DPM-Solver++, UniPC, LML (Ours)
- **分隔线**: 在LML (Ours)列后添加垂直虚线分隔
- **质量**: 300 DPI，适合论文使用

### 支持的采样器

1. **DDIM** - Denoising Diffusion Implicit Models
2. **PNDM** - Pseudo Numerical methods for Diffusion Models  
3. **DPM-Solver** - High-order solver for diffusion ODEs
4. **DPM-Solver++** - Improved version with better stability
5. **UniPC** - Unified Predictor-Corrector framework
6. **LML (Ours)** - DPM-Solver with Levenberg-Marquardt Langevin correction

## 图组特性

- **6行多列布局**：每行代表同一个生成主题，每列代表一种采样方法
- **方法标签**：顶部显示方法名称
- **分隔线**：在LML (Ours)列后添加垂直虚线分隔，突出显示你的方法
- **高质量输出**：300 DPI，适合论文使用
- **自动布局**：自动处理图像加载和布局

## 注意事项

1. **图像要求**: 需要先有对应采样器生成的图像文件
2. **目录结构**: 图像应按照 `steps{num_inference_steps}/{sampler_type}/` 结构保存
3. **文件命名**: 图像文件命名格式为 `cifar10_{sampler_type}_inference{num_inference_steps}_seed{seed}_{i}.png`
4. **依赖**: 需要安装 matplotlib 库

## 故障排除

如果遇到Qt显示错误，脚本已自动设置matplotlib使用非交互式后端（Agg），应该可以正常工作。

## 示例输出

生成的对比图组将展示：
- 每行代表同一个生成主题（同一个CIFAR-10类别）
- 每列代表一种采样方法
- LML (Ours)方法通常表现出更高的图像质量
- 在LML (Ours)列后有明显分隔线突出显示

## 与CelebA功能的区别

- **数据集**: CIFAR-10 vs CelebA-HQ
- **图像尺寸**: 32x32 vs 256x256
- **采样器**: 包含LML方法，不包含Hessian-Free方法
- **默认步数**: 20步 vs 10步
- **方法标签**: 适配CIFAR-10的采样器集合

