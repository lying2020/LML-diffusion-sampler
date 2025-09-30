# CelebA-HQ 对比图组生成功能使用说明

## 功能概述

已成功在 `celeba.py` 中添加了生成对比图组的功能，可以创建类似截图的6行多列图像对比，展示不同采样方法生成的人脸图像效果。

## 使用方法

### 1. 生成对比图组（推荐）

```bash
# 从现有图像生成对比图组
python scripts/celeba.py --generate_grid --num_inference_steps 10
```

### 2. 运行批量实验后生成对比图组

```bash
# 先运行批量实验生成图像
python scripts/celeba.py --run_batch --num_inference_steps 10

# 然后生成对比图组
python scripts/celeba.py --generate_grid --num_inference_steps 10
```

### 3. 只运行对比实验（不生成图组）

```bash
# 运行所有采样器的对比实验
python scripts/celeba.py --compare_all --num_inference_steps 10
```

## 输出结果

### 对比图组
- **文件位置**: `output/celeba/comparison_grid_steps{num_inference_steps}.png`
- **布局**: 6行7列（6行图像，7种采样方法）
- **方法顺序**: DDIM, PNDM, DPM-Solver, DPM-Solver++, UniPC, LML (Ours), HVP (Ours)
- **分隔线**: 在LML (Ours)列后添加垂直虚线分隔
- **质量**: 300 DPI，适合论文使用

### 评估结果
- **表格**: 在控制台输出详细的评估指标对比
- **日志文件**: 每个采样器的详细生成日志
- **JSON结果**: 可选的JSON格式结果文件

## 支持的采样器

1. **DDIM** - Denoising Diffusion Implicit Models
2. **PNDM** - Pseudo Numerical methods for Diffusion Models  
3. **DPM-Solver** - High-order solver for diffusion ODEs
4. **DPM-Solver++** - Improved version with better stability
5. **UniPC** - Unified Predictor-Corrector framework
6. **LML (Ours)** - DPM-Solver with Levenberg-Marquardt Langevin correction
7. **HVP (Ours)** - DPM-Solver with Hessian-Free HVP correction using CG

## 评估指标

- **ColorS** - Colorfulness score
- **FS** - Face Score (simplified)
- **DFIQA** - Deep Face Image Quality Assessment
- **PicS** - Picture Score (aesthetic quality)
- **EAT** - Enhanced Aesthetic Test
- **Laion** - LAION aesthetic score

## 注意事项

1. **图像要求**: 需要先有对应采样器生成的图像文件
2. **目录结构**: 图像应按照 `steps{num_inference_steps}/{sampler_type}/` 结构保存
3. **文件命名**: 图像文件命名格式为 `celeba_{sampler_type}_inference{num_inference_steps}_seed{seed}_{i}.png`
4. **依赖**: 需要安装 matplotlib 库

## 故障排除

如果遇到Qt显示错误，脚本已自动设置matplotlib使用非交互式后端（Agg），应该可以正常工作。

## 示例输出

生成的对比图组将展示：
- 每行代表同一个生成主题（同一个人脸）
- 每列代表一种采样方法
- LML (Ours)和HVP (Ours)方法通常表现出更高的图像质量
- 在LML (Ours)列后有明显分隔线突出显示

