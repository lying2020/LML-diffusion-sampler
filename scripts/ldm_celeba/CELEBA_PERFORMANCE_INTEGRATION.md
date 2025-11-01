# CelebA性能分析功能集成说明

## 概述

已成功将 `celeba_performance.py` 中的关键功能集成到 `celeba.py` 中，现在可以在一个统一的脚本中进行图像生成和性能分析。

## 新增功能

- **ColorS**: 颜色丰富度
- **FS**: 人脸质量分数
- **DFIQA**: 深度人脸图像质量评估
- **PicS**: 图片美学质量
- **EAT**: 增强美学测试
- **Laion**: LAION美学分数


### 1. 性能数据分析器类 (`CelebAPerformanceAnalyzer`)

- **自动查找JSON日志文件**：扫描输出目录中的所有生成日志
- **性能数据提取**：从JSON文件中提取详细的性能统计信息
- **数据聚合分析**：按步数和方法分组进行统计分析
- **多格式报告生成**：支持CSV、LaTeX表格输出

### 2. 新增命令行参数

```bash
# 性能分析选项
--analyze_performance          # 在实验结束后运行性能分析
--performance_output_dir       # 性能分析输出目录 (默认: output/celeba)
--performance_only            # 仅运行性能分析，不进行图像生成
```

### 3. 增强的批量实验模式

- **自动性能分析**：批量实验完成后可自动进行性能分析
- **性能摘要生成**：在实验总结中包含性能统计信息
- **多格式输出**：生成CSV、LaTeX格式的性能对比表格

## 使用方法

### 1. 运行实验并自动分析性能

```bash
# 运行批量实验并自动进行性能分析
python scripts/celeba.py --run_batch --analyze_performance

# 指定性能分析输出目录
python scripts/celeba.py --run_batch --analyze_performance --performance_output_dir output/celeba_ldm
```

### 2. 仅运行性能分析（分析已有结果）

```bash
# 分析已有的实验结果
python scripts/celeba.py --performance_only --performance_output_dir output/celeba_ldm
```

### 3. 单个实验模式（保持原有功能）

```bash
# 运行单个实验（不包含性能分析）
python scripts/celeba.py --sampler_type  --num_inference_steps 20
```

## 输出文件

### 性能分析生成的文件：

1. **原始数据**：`celeba_performance_data_YYYYMMDD_HHMMSS.csv`
2. **时间对比表**：`celeba_time_comparison_YYYYMMDD_HHMMSS.csv`
3. **速度对比表**：`celeba_ips_comparison_YYYYMMDD_HHMMSS.csv`
4. **LaTeX时间表格**：`celeba_time_latex_YYYYMMDD_HHMMSS.tex`
5. **LaTeX速度表格**：`celeba_ips_latex_YYYYMMDD_HHMMSS.tex`
6. **性能摘要**：`performance_summary.json`

## 性能分析功能

### 1. 数据提取
- 自动扫描 `steps_*` 目录下的JSON日志文件
- 提取生成时间、速度、图片数量等关键指标
- 支持多种采样方法的性能对比

### 2. 统计分析
- 按步数和方法分组的性能统计
- 计算平均值、标准差、最小值、最大值
- 识别最快和最慢的采样方法

### 3. 报告生成
- 控制台输出性能摘要表格
- 生成CSV格式的详细数据
- 创建LaTeX格式的学术论文表格
- 保存JSON格式的性能摘要

## 集成优势

1. **统一接口**：一个脚本完成图像生成和性能分析
2. **自动化流程**：批量实验后自动进行性能分析
3. **灵活使用**：可以单独运行性能分析或与实验结合
4. **多格式输出**：支持多种数据格式，便于后续分析
5. **详细日志**：完整的日志记录和错误处理

## 注意事项

- 性能分析需要先有JSON格式的生成日志文件
- 确保 `pandas` 库已安装（用于数据处理）
- LaTeX表格生成需要相应的LaTeX环境（可选）
- 性能分析会扫描指定目录下的所有相关JSON文件

## 示例输出

```
📋 性能摘要表格:
================================================================================

步数 10:
----------------------------------------
  ddim        : 0.1234s/img, 8.10 img/s, 20 images
  : 0.1456s/img, 6.87 img/s, 20 images
  pndm        : 0.1345s/img, 7.43 img/s, 20 images

步数 20:
----------------------------------------
  ddim        : 0.2345s/img, 4.26 img/s, 20 images
  : 0.2567s/img, 3.90 img/s, 20 images
  pndm        : 0.2456s/img, 4.07 img/s, 20 images

📈 统计信息:
==================================================
总实验数: 6
步数范围: 10 - 20
方法数: 3
总生成图片数: 120
总耗时: 28.45 秒

🚀 最快方法: ddim (步数10) - 8.10 img/s
🐌 最慢方法:  (步数20) - 3.90 img/s
```
