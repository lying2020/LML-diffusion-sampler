# CIFAR-10 统一采样脚本使用说明

## 📋 概述

`cifar10.py` 是一个统一的CIFAR-10图像生成脚本，整合了原来 `cifar10.py` 和 `cifar10_updated.py` 的功能，并增加了许多增强特性。

## 🚀 主要特性

### ✨ 增强功能
- **统一接口**: 整合了两个原始脚本的功能
- **详细日志**: 可选的生成日志记录
- **进度显示**: 实时显示生成进度
- **性能统计**: 自动计算生成时间和效率
- **错误处理**: 完善的错误处理和调试信息
- **灵活配置**: 支持多种参数组合

### 🎯 支持的采样器
- `ddim`: Denoising Diffusion Implicit Models
- `dpm`: DPM-Solver
- `dpm++`: DPM-Solver++ (改进版)
- `dpm_lm`: DPM-Solver with LML correction
- `pndm`: Pseudo Numerical methods
- `unipc`: Unified Predictor-Corrector

## 📖 使用方法

### 基本用法
```bash
# 使用默认设置生成图像
python scripts/cifar10.py --sampler_type dpm_lm --test_num 10

# 生成DDIM图像
python scripts/cifar10.py --sampler_type ddim --test_num 5

# 使用不同的输出目录
python scripts/cifar10.py --sampler_type dpm++ --save_dir ./output/my_results
```

### 高级用法
```bash
# 使用自定义LML参数
python scripts/cifar10.py --sampler_type dpm_lm --lamb 0.001 --kappa 1e-7 --test_num 20

# 使用半精度浮点数加速
python scripts/cifar10.py --sampler_type ddim --dtype fp16 --test_num 10

# 生成大量图像并保存日志
python scripts/cifar10.py --sampler_type dpm_lm --test_num 50 --batch_size 8 --save_log

# 启用详细输出模式
python scripts/cifar10.py --sampler_type dpm++ --test_num 10 --verbose
```

## ⚙️ 参数说明

### 基本参数
- `--test_num`: 生成批次数量 (默认: 1)
- `--start_index`: 起始种子索引 (默认: 0)
- `--batch_size`: 每批次图像数量 (默认: 4)
- `--num_inference_steps`: 去噪步数 (默认: 20)

### 采样器参数
- `--sampler_type`: 采样器类型 (默认: dpm_lm)
- `--lamb`: LML校正的lambda参数 (默认: 0.0008)
- `--kappa`: EMA平滑的kappa参数 (默认: 1.0e-8)

### 输出参数
- `--save_dir`: 输出目录 (默认: ./output/test/cifar10)
- `--model_id`: 模型路径 (默认: ddpm_ema_cifar10)

### 技术参数
- `--dtype`: 数据类型 (fp32/fp64/fp16/bf16, 默认: fp32)
- `--device`: 计算设备 (默认: cuda)

### 可选功能
- `--save_log`: 保存生成日志到JSON文件
- `--verbose`: 启用详细输出

## 📊 输出说明

### 生成的文件
- **图像文件**: `cifar10_{sampler_type}_inference{steps}_seed{seed}_{index}.png`
- **日志文件**: `generation_log_{sampler_type}_{timestamp}.json` (如果使用--save_log)

### 控制台输出
- 实时进度显示
- 性能统计信息
- 错误和警告信息
- 生成总结

## 🔧 配置示例

### 快速测试
```bash
python scripts/cifar10.py --sampler_type ddim --test_num 2 --batch_size 2
```

### 高质量生成
```bash
python scripts/cifar10.py --sampler_type dpm_lm --test_num 20 --num_inference_steps 50 --lamb 0.001
```

### 批量生成
```bash
python scripts/cifar10.py --sampler_type dpm++ --test_num 100 --batch_size 8 --save_log
```

### 性能测试
```bash
python scripts/cifar10.py --sampler_type ddim --test_num 50 --dtype fp16 --verbose
```

## 📈 性能优化建议

### 速度优化
- 使用 `--dtype fp16` 减少内存使用
- 减少 `--num_inference_steps` 步数
- 增加 `--batch_size` 批量大小

### 质量优化
- 增加 `--num_inference_steps` 步数
- 调整 `--lamb` 和 `--kappa` 参数
- 使用 `dpm_lm` 或 `dpm++` 采样器

### 内存优化
- 使用 `--dtype fp16` 或 `bf16`
- 减少 `--batch_size`
- 使用 `--device cpu` (如果GPU内存不足)

## 🐛 故障排除

### 常见问题
1. **模型路径错误**: 确保模型文件在 `model/ddpm_ema_cifar10/` 目录下
2. **CUDA内存不足**: 减少 `--batch_size` 或使用 `--dtype fp16`
3. **权限错误**: 确保对输出目录有写权限

### 调试模式
```bash
python scripts/cifar10.py --sampler_type ddim --test_num 1 --verbose
```

## 📝 日志文件格式

生成的JSON日志文件包含：
- 时间戳和参数信息
- 生成统计信息
- 性能指标
- 错误信息（如果有）

## 🔄 与原脚本的兼容性

新的统一脚本完全兼容原来的使用方式：
- 默认参数与原脚本相同
- 支持所有原有的采样器类型
- 保持相同的输出格式

## 📚 更多信息

- 查看 `--help` 获取完整参数列表
- 使用 `--verbose` 获取详细调试信息
- 检查生成的日志文件了解性能统计

---

**版本**: 统一版本 v1.0
**更新日期**: 2025年9月15日
**兼容性**: 完全向后兼容
