# FID Score 优化说明

## 优化内容

已成功优化 `evaluations/fid_score_new.py` 文件，添加了对大量图片的处理限制，提高计算效率。

## 主要改进

### 1. 图片数量限制
- **默认限制**: 每个方法最多处理 10,000 张图片
- **随机采样**: 当图片数量超过限制时，使用随机采样选择代表性子集
- **可配置**: 通过 `--max-images` 参数自定义最大图片数量

### 2. 性能优化
- **内存效率**: 避免处理过多图片导致内存不足
- **计算速度**: 减少不必要的计算时间
- **可重现性**: 使用固定随机种子确保结果可重现

### 3. 新增功能
- **智能采样**: 当图片数量超过限制时自动采样
- **进度提示**: 显示实际处理的图片数量
- **缓存支持**: 保持原有的缓存功能

## 使用方法

### 基本用法
```bash
# 使用默认设置（最多10000张图片）
python evaluations/fid_score_new.py path1 path2

# 自定义最大图片数量
python evaluations/fid_score_new.py --max-images 5000 path1 path2

# 批量模式
python evaluations/fid_score_new.py --batch-mode --max-images 8000 --real-path data/real --output-path output/generated
```

### 参数说明
- `--max-images`: 最大图片数量限制（默认: 10000）
- `--batch-mode`: 批量处理模式
- `--real-path`: 真实图片路径
- `--output-path`: 生成图片路径
- `--use-cache`: 使用缓存（默认: True）

## 优化效果

### 处理大量图片时
- **之前**: 处理所有图片，可能导致内存不足或计算时间过长
- **现在**: 自动限制图片数量，确保稳定高效的计算

### 采样策略
- 使用随机采样选择代表性子集
- 保持图片的多样性
- 确保结果的可重现性

### 性能提升
- 减少内存使用
- 提高计算速度
- 避免系统崩溃

## 示例输出

```
Found 25000 images, limiting to 10000 for efficiency
Computing statistics for 10000 images in /path/to/images
```

## 注意事项

1. **随机种子**: 使用固定种子（42）确保结果可重现
2. **采样顺序**: 采样后保持排序顺序
3. **缓存兼容**: 与原有缓存系统完全兼容
4. **向后兼容**: 保持原有API不变

## 技术细节

### 采样算法
```python
if len(files) > max_images:
    import random
    random.seed(42)  # 确保可重现性
    files = random.sample(files, max_images)
    files.sort()  # 保持排序
```

### 函数更新
- `get_activations()`: 添加 `max_images` 参数
- `calculate_activation_statistics()`: 支持图片数量限制
- `compute_statistics_of_path()`: 智能采样处理
- `batch_calculate_fid()`: 批量处理优化

## 测试建议

1. **小数据集**: 测试原有功能是否正常
2. **大数据集**: 验证采样功能是否有效
3. **性能测试**: 对比优化前后的计算时间
4. **结果验证**: 确保采样结果具有代表性

