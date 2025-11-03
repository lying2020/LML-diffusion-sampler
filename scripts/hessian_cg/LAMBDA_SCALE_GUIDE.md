# Lambda Scale 参数使用指南

## 一、背景

在 HCG 方法中，`lambda_t` 是自适应计算的阻尼参数：

```
lambda_t_base = max(0, (α_t - κ_* * β_t) / (κ_* - 1))
lambda_t = lambda_scale * lambda_t_base
```

虽然 `lambda_t` 是自适应的，但有时我们需要对其进行缩放以更好地匹配特定数据集或模型的特性。

## 二、为什么需要 lambda_scale？

### 2.1 LML vs HCG 的对比

**LML 方法**：
- 使用固定的 `λ` 参数（通常 0.0001 - 0.01）
- 需要手动调优

**HCG 方法**：
- 自适应计算 `lambda_t`，但可能：
  - 过大：过阻尼，收敛慢
  - 过小：阻尼不足，数值不稳定
- 需要 `lambda_scale` 来精细调节

### 2.2 lambda_t 的典型取值范围

根据实验观察（CelebA-HQ, 50步采样）：

| 阶段 | alpha_t 范围 | beta_t 范围 | kappa_current | lambda_t_base 范围 |
|------|-------------|-------------|---------------|-------------------|
| 早期 (t=900-800) | 0.8-1.0 | 0.05-0.1 | 10-20 | 0.000-0.005 |
| 中期 (t=500-300) | 0.5-0.8 | 0.01-0.05 | 15-50 | 0.002-0.015 |
| 后期 (t=100-50) | 0.2-0.5 | 0.001-0.01 | 50-200 | 0.005-0.030 |

**典型范围**：`lambda_t_base` 通常在 **0.001 - 0.03** 之间。

## 三、如何选择合适的 lambda_scale？

### 3.1 方法 1：启用统计日志

首先，运行采样并启用 `--log_lambda_stats` 来查看实际的 `lambda_t` 值：

```bash
python scripts/ldm_celeba/celeba.py \
    --sampler_type dpm_hcg \
    --num_inference_steps 50 \
    --test_num 1 \
    --log_lambda_stats \
    --lambda_scale 1.0
```

输出示例：
```
[HCG lambda stats] t=999, alpha_t=0.952341, beta_t=0.089123, kappa_current=10.68, lambda_t_base=0.004521, lambda_t_scaled=0.004521, lambda_scale=1.0000
[HCG lambda stats] t=980, alpha_t=0.891234, beta_t=0.067891, kappa_current=13.14, lambda_t_base=0.007234, lambda_t_scaled=0.007234, lambda_scale=1.0000
...
```

观察 `lambda_t_base` 的范围，然后：
- 如果平均约为 0.01，想要 0.004（类似 LML）：`lambda_scale = 0.4`
- 如果想要更强的阻尼：`lambda_scale > 1.0`
- 如果想要更弱的阻尼：`lambda_scale < 1.0`

### 3.2 方法 2：基于 LML 的经验值

如果你知道 LML 方法在你的数据集上的最优 `λ` 值：

```
lambda_scale = target_lambda / average_lambda_t_base
```

例如：
- LML 最优 `λ = 0.004`
- HCG 平均 `lambda_t_base = 0.01`
- 则 `lambda_scale = 0.004 / 0.01 = 0.4`

### 3.3 方法 3：网格搜索

对于不同数据集，可能需要不同的 `lambda_scale`：

```bash
# 测试不同的 lambda_scale 值
for scale in 0.2 0.4 0.6 0.8 1.0 1.2 1.5 2.0; do
    python scripts/ldm_celeba/celeba.py \
        --sampler_type dpm_hcg \
        --lambda_scale $scale \
        --test_num 10
done
```

然后根据 FID 或其他质量指标选择最优值。

## 四、典型使用场景

### 4.1 CelebA-HQ (推荐设置)

```bash
--lambda_scale 0.4  # 相当于 LML 的 λ ≈ 0.004
```

### 4.2 CIFAR-10 (推荐设置)

```bash
--lambda_scale 0.3  # 相当于 LML 的 λ ≈ 0.0008 (20步)
```

### 4.3 默认设置

```bash
--lambda_scale 1.0  # 使用理论值，不做缩放
```

## 五、lambda_scale 的影响

### 5.1 对性能的影响

| lambda_scale | 效果 | 适用场景 |
|-------------|------|----------|
| < 0.5 | 弱阻尼，可能不稳定 | 各向同性阶段，快速采样 |
| 0.5 - 1.0 | 中等阻尼（推荐） | 大多数场景 |
| 1.0 - 2.0 | 强阻尼，更稳定 | 各向异性阶段 |
| > 2.0 | 过阻尼，可能过慢 | 特殊情况 |

### 5.2 与 LML 的对应关系

| LML lambda | HCG lambda_scale (假设 base=0.01) | HCG lambda_t 实际值 |
|-----------|----------------------------------|-------------------|
| 0.0001 | 0.01 | 0.0001 |
| 0.0008 | 0.08 | 0.0008 |
| 0.004 | 0.4 | 0.004 |
| 0.01 | 1.0 | 0.01 |

## 六、实验建议

### 6.1 首次使用

1. 运行一次采样，启用 `--log_lambda_stats`
2. 观察 `lambda_t_base` 的典型范围
3. 根据目标值计算 `lambda_scale`
4. 重新运行并比较结果

### 6.2 调优流程

```
1. 使用默认 lambda_scale=1.0 运行
   ↓
2. 启用 log_lambda_stats 查看范围
   ↓
3. 根据 LML 经验或目标值计算 scale
   ↓
4. 微调 scale (0.1 步长)
   ↓
5. 选择 FID/质量最好的值
```

## 七、注意事项

1. **lambda_scale 不是越大越好**：过大的 scale 会导致过阻尼，采样质量可能下降
2. **与 kappa_star 的交互**：调整 `kappa_star` 也会影响 `lambda_t_base` 的范围
3. **不同步数的差异**：不同采样步数可能需要不同的 `lambda_scale`
4. **数据集相关**：不同数据集的最优 `lambda_scale` 可能不同

## 八、示例命令

```bash
# 查看 lambda_t 统计信息
python scripts/ldm_celeba/celeba.py \
    --sampler_type dpm_hcg \
    --num_inference_steps 50 \
    --test_num 1 \
    --log_lambda_stats

# 使用推荐的 lambda_scale (类似 LML λ=0.004)
python scripts/ldm_celeba/celeba.py \
    --sampler_type dpm_hcg \
    --num_inference_steps 50 \
    --lambda_scale 0.4

# 测试不同的 lambda_scale
python scripts/ldm_celeba/celeba.py \
    --sampler_type dpm_hcg \
    --num_inference_steps 50 \
    --lambda_scale 0.3  # 试试更小的值
```
