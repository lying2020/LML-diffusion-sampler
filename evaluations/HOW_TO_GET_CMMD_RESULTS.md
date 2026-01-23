# 如何获取 CMMD 评测结果

## 问题说明

如果你看到的结果中只有 CLIP-Score 而没有 CMMD，可能是因为：

1. **使用了错误的参数**：使用了 `--batch_mode` 而不是 `--batch_all_metrics`
2. **CMMD 库未安装**：`cmmd-pytorch` 库不可用
3. **参考目录问题**：参考图像目录不存在或为空

## 解决方案

### 方法 1: 使用正确的参数（推荐）

要同时获取 CLIP-Score 和 CMMD 结果，需要使用 `--batch_all_metrics` 参数：

```bash
python evaluations/cmmd_clipscore_eval.py \
    --output_dir output/coco_stable-diffusion-2-base \
    --prompts_file evaluations/coco_prompts/coco_top_40_prompts_backup.json \
    --batch_all_metrics \
    --output_json output/coco_stable-diffusion-2-base/evaluation_results_with_cmmd.json
```

**区别**：
- `--batch_mode`: 只计算 CLIP-Score（更快）
- `--batch_all_metrics`: 计算 CLIP-Score + CMMD（需要参考图像）

### 方法 2: 检查 CMMD 库

如果使用了 `--batch_all_metrics` 但仍然没有 CMMD 结果，检查 CMMD 库：

```bash
python3 -c "from cmmd_pytorch import compute_cmmd; print('CMMD library available')"
```

如果报错，说明库未安装。由于 `cmmd-pytorch` 可能无法从 PyPI 安装，可以：
1. 尝试从 GitHub 安装（如果存在）
2. 或者只使用 CLIP-Score 结果

### 方法 3: 检查参考图像目录

确保参考图像目录存在且包含图像：

```bash
ls /home/liying/Documents/dataset/coco/val2014 | head -10
```

如果目录不存在或为空，CMMD 将无法计算。

## 查看已有结果中的 CMMD

如果你想从已有的 JSON 结果文件中查看是否有 CMMD 数据：

```bash
python3 -c "
import json
data = json.load(open('output/coco_stable-diffusion-2-base/evaluation_results.json'))
# 检查是否有 CMMD
has_cmmd = any('cmmd' in str(v) for v in data.get('results', {}).values())
print('Has CMMD data:', has_cmmd)
if has_cmmd:
    # 显示第一个有 CMMD 的结果
    for k, v in data.get('results', {}).items():
        if 'cmmd' in v:
            print(f'{k}: CMMD = {v[\"cmmd\"][\"score\"]:.4f}')
            break
"
```

## 重新运行以获取 CMMD 结果

如果你之前使用了 `--batch_mode`，需要重新运行：

```bash
# 重新运行，使用 --batch_all_metrics
python evaluations/cmmd_clipscore_eval.py \
    --output_dir output/coco_stable-diffusion-2-base \
    --prompts_file evaluations/coco_prompts/coco_top_40_prompts_backup.json \
    --batch_all_metrics \
    --reference_dir /home/liying/Documents/dataset/coco/val2014 \
    --output_json output/coco_stable-diffusion-2-base/evaluation_results_with_cmmd.json
```

这次运行会：
1. 计算所有目录的 CLIP-Score
2. 计算所有目录的 CMMD（如果库可用且参考图像存在）
3. 在汇总表格中显示两列：CLIP-Score 和 CMMD

## 输出格式

使用 `--batch_all_metrics` 后，输出表格会包含 CMMD 列：

```
Path                                          Steps    Sampler    CLIP-Score          CMMD        Images
----------------------------------------------------------------------------------------------------------
steps_10/dpm_lm                               10      dpm_lm     26.2557 ± 1.8472     0.1234      20
steps_10/dpm++                                10      dpm++      26.0733 ± 3.2363     0.1256      20
...
```

如果 CMMD 不可用，表格中会显示 "N/A"。
