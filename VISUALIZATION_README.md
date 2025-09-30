# CelebA-HQ 对比图组生成功能

## 功能描述

在 `celeba.py` 中新增了两个函数来生成类似截图的对比图组，展示不同采样方法生成的人脸图像效果。

## 新增函数

### 1. `generate_comparison_grid(save_dir, sampler_types, num_inference_steps=10, test_num=6, batch_size=1)`

生成对比图组的主要函数。

**参数：**
- `save_dir`: 图像保存目录
- `sampler_types`: 采样器类型列表，如 `['ddim', 'pndm', 'dpm', 'dpm++', 'unipc', 'dpm_lm', 'hessian_free']`
- `num_inference_steps`: 推理步数，默认10
- `test_num`: 行数（图像数量），默认6
- `batch_size`: 批次大小，默认1

**功能：**
- 创建6行多列的图像网格
- 每列代表一种采样方法
- 每行代表同一个主题的不同生成结果
- 自动添加方法标签和分隔线
- 支持LML方法后的虚线分隔

### 2. `generate_comparison_grid_from_existing(save_dir, num_inference_steps=10)`

从已存在的图像文件生成对比图组。

**参数：**
- `save_dir`: 图像保存目录
- `num_inference_steps`: 推理步数，默认10

**功能：**
- 自动检测可用的采样器
- 从现有图像文件生成对比图组
- 无需重新生成图像

## 使用方法

### 方法1：命令行使用

```bash
# 生成对比图组（需要先有图像文件）
python scripts/celeba.py --generate_grid --num_inference_steps 10

# 运行批量实验后生成对比图组
python scripts/celeba.py --run_batch --num_inference_steps 10
python scripts/celeba.py --generate_grid --num_inference_steps 10
```

### 方法2：Python代码调用

```python
from scripts.celeba import generate_comparison_grid_from_existing
import project as project

# 生成对比图组
save_dir = os.path.join(project.output_dir, 'celeba')
output_path = generate_comparison_grid_from_existing(save_dir, num_inference_steps=10)
print(f"对比图组已保存到: {output_path}")
```

### 方法3：测试脚本

```bash
# 运行测试脚本
python test_visualization.py
```

## 输出结果

生成的对比图组将保存为 `comparison_grid_steps{num_inference_steps}.png`，包含：

- **6行图像**：每行代表同一个生成主题
- **多列方法**：每列代表一种采样方法
- **方法标签**：顶部显示方法名称（DDIM, PNDM, DPM-Solver, DPM-Solver++, UniPC, LML (Ours), HVP (Ours)）
- **分隔线**：在LML方法后添加垂直虚线分隔
- **高质量输出**：300 DPI，适合论文使用

## 支持的采样器

- DDIM
- PNDM  
- DPM-Solver
- DPM-Solver++
- UniPC
- LML (Ours) - DPM-Solver with Levenberg-Marquardt Langevin correction
- HVP (Ours) - DPM-Solver with Hessian-Free HVP correction

## 注意事项

1. 需要先运行批量实验生成图像文件
2. 图像文件应按照 `steps{num_inference_steps}/{sampler_type}/` 的目录结构保存
3. 图像文件命名格式：`celeba_{sampler_type}_inference{num_inference_steps}_seed{seed}_{i}.png`
4. 需要安装 matplotlib 库：`pip install matplotlib`

## 示例输出

生成的对比图组将类似以下布局：

```
        DDIM    PNDM    DPM-Solver    DPM-Solver++    UniPC    LML (Ours)    HVP (Ours)
Row 1   [img]   [img]   [img]         [img]           [img]    [img] |       [img]
Row 2   [img]   [img]   [img]         [img]           [img]    [img] |       [img]
Row 3   [img]   [img]   [img]         [img]           [img]    [img] |       [img]
Row 4   [img]   [img]   [img]         [img]           [img]    [img] |       [img]
Row 5   [img]   [img]   [img]         [img]           [img]    [img] |       [img]
Row 6   [img]   [img]   [img]         [img]           [img]    [img] |       [img]
```

其中 `|` 表示分隔线。
