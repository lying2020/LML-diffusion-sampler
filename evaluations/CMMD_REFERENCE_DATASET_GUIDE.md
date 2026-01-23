# CMMD 评测参考数据集指南

## 关于 Stable Diffusion 2.0 的训练数据

**Stable Diffusion 2.0 并不是基于 MS COCO 数据集微调的**。实际情况是：

- **主要训练数据**: Stable Diffusion 2.0 主要在 **LAION-5B** 数据集上训练
- **LAION-5B**: 一个包含 50 亿图像-文本对的大规模数据集
- **MS COCO**: 主要用于评测和验证，不是训练数据

## CMMD 评测指标需要什么样的参考数据集？

### CMMD (Conditional Maximum Mean Discrepancy) 原理

CMMD 用于衡量**生成图像分布**和**真实图像分布**之间的距离。它需要：

1. **参考图像集（Real Images）**: 来自真实数据集的图像
2. **生成图像集（Generated Images）**: 由你的模型生成的图像

### 参考数据集的选择原则

#### ✅ 推荐方案：使用 MS COCO 验证集/测试集

由于你使用的是 COCO prompts (`coco_top_40_prompts_backup.json`)，**最佳选择是使用 MS COCO 数据集的真实图像作为参考**。

**原因**：
- 你的生成图像是基于 COCO prompts 的
- 使用相同领域的真实图像（COCO）可以更准确地评估生成质量
- COCO 数据集是标准评测基准，结果具有可比性

#### 📊 数据集要求

1. **图像数量**:
   - 建议至少 **1000+ 张图像**（越多越好，通常使用 5000-10000 张）
   - 如果只有 20 个 prompts，可以使用 COCO 验证集中对应的图像

2. **图像质量**:
   - 高分辨率（通常 512x512 或更高）
   - 清晰的图像，无损坏

3. **图像来源**:
   - MS COCO 2017 验证集（约 5,000 张图像）
   - MS COCO 2017 测试集（约 40,000 张图像）
   - 或者从 COCO 数据集中选择与你的 prompts 相关的图像

## 如何准备参考数据集

### 方案 1: 下载 MS COCO 验证集（推荐）

```bash
# 下载 COCO 2017 验证集
# 可以从官方下载：https://cocodataset.org/#download
# 或者使用 Python 脚本下载
```

### 方案 2: 从 COCO 数据集中提取与你的 prompts 对应的图像

如果你有 COCO 数据集的标注文件，可以：
1. 根据你的 prompts 找到对应的 COCO 图像 ID
2. 提取这些图像作为参考数据集

### 方案 3: 使用通用的高质量图像数据集

如果无法获取 COCO 图像，可以使用：
- **LAION-5B 子集**: 与训练数据分布相似
- **ImageNet**: 高质量的自然图像
- **其他公开数据集**: 确保图像质量和多样性

## 使用建议

### 对于你的情况（使用 COCO prompts）

**最佳实践**：

1. **下载 MS COCO 2017 验证集**
   - 包含约 5,000 张图像
   - 覆盖多种场景和对象

2. **或者提取与你的 prompts 对应的 COCO 图像**
   - 如果你知道每个 prompt 对应的 COCO 图像 ID
   - 提取这些特定图像作为参考

3. **评测时使用**：
   ```bash
   # Using default reference directory
   python evaluations/cmmd_clipscore_eval.py \
       --output_dir output/coco_stable-diffusion-2-base/steps_20/dpm_lm \
       --prompts_file evaluations/coco_prompts/coco_top_40_prompts_backup.json \
       --clip_model /home/liying/Documents/clip-vit-large-patch14

   # Or specify a different reference directory
   python evaluations/cmmd_clipscore_eval.py \
       --output_dir output/coco_stable-diffusion-2-base/steps_20/dpm_lm \
       --prompts_file evaluations/coco_prompts/coco_top_40_prompts_backup.json \
       --reference_dir /path/to/coco/val2017 \
       --clip_model /home/liying/Documents/clip-vit-large-patch14
   ```

   **Default reference directory**: `/home/liying/Documents/dataset/coco/val2014`

## 注意事项

1. **数据分布匹配**: 参考数据集应该与生成图像的分布尽可能匹配
2. **图像数量**: 更多参考图像通常能得到更稳定的 CMMD 分数
3. **图像预处理**: 确保参考图像和生成图像使用相同的预处理方式（在代码中已处理）

## 下载 MS COCO 数据集

### 官方下载链接
- **COCO 2017 验证集**: http://images.cocodataset.org/zips/val2017.zip
- **COCO 2017 测试集**: http://images.cocodataset.org/zips/test2017.zip
- **标注文件**: http://images.cocodataset.org/annotations/annotations_trainval2017.zip

### 使用 Python 下载（示例）

```python
import urllib.request
import zipfile
import os

# 下载验证集
url = "http://images.cocodataset.org/zips/val2017.zip"
output_dir = "data/coco"
os.makedirs(output_dir, exist_ok=True)

print("Downloading COCO validation set...")
urllib.request.urlretrieve(url, os.path.join(output_dir, "val2017.zip"))

print("Extracting...")
with zipfile.ZipFile(os.path.join(output_dir, "val2017.zip"), 'r') as zip_ref:
    zip_ref.extractall(output_dir)

print("Done! Images are in:", os.path.join(output_dir, "val2017"))
```

## 总结

对于你的 Stable Diffusion 2.0 + COCO prompts 场景：

✅ **推荐**: 使用 MS COCO 2017 验证集作为参考数据集
- 下载地址: http://images.cocodataset.org/zips/val2017.zip
- 包含约 5,000 张高质量图像
- 与你的生成图像领域匹配

❌ **不推荐**: 使用与 COCO 不相关的数据集（如 ImageNet、CelebA 等）
- 分布不匹配会导致 CMMD 分数不准确
