# FID计算脚本使用说明

## 文件说明

- `fid_compute.sh`: 主要的FID计算脚本
- `fid_summary.py`: 统计报告生成脚本

## 使用方法

### 1. 基本用法

```bash
./fid_compute.sh <cifar10_real_images_path> <generated_images_path>
```

### 2. 示例

```bash
./fid_compute.sh data/cifar10/cifar10_images output/cifar10
```

## 脚本功能

### fid_compute.sh

1. **参数检查**: 验证输入参数是否正确
2. **依赖检查**: 检查Python和pytorch-fid是否安装
3. **路径检查**: 验证输入路径是否存在
4. **方法发现**: 自动发现所有可用的生成方法
5. **FID计算**: 为每个方法和步数组合计算FID值
6. **结果输出**: 生成表格格式和CSV格式的结果文件

### fid_summary.py

1. **结果解析**: 解析FID结果文件
2. **统计分析**: 计算各种统计指标
3. **报告生成**: 生成详细的统计报告

## 输出文件

运行脚本后会生成以下文件：

- `fid_results_table.txt`: 表格格式的FID结果
- `fid_results_table.csv`: CSV格式的FID结果
- `fid_results_table_summary.txt`: 详细的统计报告

## 支持的步数

脚本支持以下步数：5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 80

## 目录结构要求

生成图片的目录结构应该是：
```
output/cifar10/
├── steps_5/
│   ├── method1/
│   ├── method2/
│   └── ...
├── steps_6/
│   ├── method1/
│   ├── method2/
│   └── ...
└── ...
```

## 注意事项

1. 确保真实图片路径包含PNG格式的图片
2. 确保生成图片路径包含PNG格式的图片
3. 脚本会自动安装pytorch-fid依赖（如果未安装）
4. 如果某个方法或步数的图片不存在，会在结果中显示"N/A"

## 错误处理

脚本包含完整的错误处理机制：
- 参数验证
- 路径存在性检查
- 依赖检查
- 图片文件检查
- FID计算错误处理

## 颜色输出

脚本使用颜色输出便于查看：
- 蓝色: 信息提示
- 绿色: 成功消息
- 黄色: 警告消息
- 红色: 错误消息
