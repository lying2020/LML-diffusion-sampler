# HCG vs LML 方法对比与问题排查

## 核心差异对比

### LML 方法 (`lm_correct`)
```python
# 1. EMA 平滑
noise_pred_ema = kappa * prev_noise + (1 - kappa) * noise_pred

# 2. 显式公式（单步计算，无迭代）
inner_product = sum(noise_pred * noise_pred_ema)
part2 = noise_pred_ema * inner_product / (lamb + norm_squared_ema)
inversed_pred = noise_pred - part2

# 3. 归一化方向（保持原始幅度）
inversed_pred = inversed_pred * norm_original / norm_corrected
```

**特点**：
- ✅ **显式公式**：直接计算，无迭代误差
- ✅ **EMA 平滑**：利用历史信息，减少噪声
- ✅ **固定 lambda**：0.004，简单可控
- ✅ **归一化**：保持原始噪声幅度

### HCG 方法 (`hcg_correct`)
```python
# 1. 估计特征值（Lanczos，多次 HVP）
alpha_t, beta_t = lanczos_eigenvalue_estimation(...)

# 2. 自适应 lambda_t
lambda_t = adaptive_damping_lambda(alpha_t, beta_t, kappa_star) * lambda_scale

# 3. CG 迭代求解 (H + lambda_t I)^{-1} * noise_pred
corrected_noise = conjugate_gradient_solve(noise_pred, ...)

# 4. c_t 缩放
corrected_noise = c_t * corrected_noise  # c_t = beta_t + lambda_t (或 1.0)

# 5. 归一化（保持原始幅度）
corrected_noise = corrected_noise * norm_original / norm_corrected
```

**特点**：
- ❌ **迭代求解**：CG 可能有收敛误差
- ❌ **无 EMA 平滑**：没有利用历史信息
- ⚠️ **自适应 lambda_t**：可能不合适
- ⚠️ **c_t 缩放**：可能过小，削弱校正效果

## 潜在问题点

### 问题 1: CG 求解不准确
- **症状**：CG 迭代次数太少（平均 0.14），几乎不执行
- **可能原因**：
  - `cg_tol` 太松（即使改成 1e-3 也可能不够）
  - `cg_max_iter` 太少（5 次可能不够）
  - 初始 guess（warm start）太接近解，但实际不准确

### 问题 2: c_t 缩放过小
- **症状**：`c_t = beta_t + lambda_t ≈ 0.00001 + 0.01 ≈ 0.01`，缩放后校正几乎消失
- **可能原因**：
  - `beta_t` 太小（接近下界 1e-5）
  - `lambda_t` 太小
  - 即使归一化，也可能无法恢复正确的校正方向

### 问题 3: lambda_t 不合适
- **症状**：条件数控制失败，CG 收敛困难
- **可能原因**：
  - 自适应公式在某些 timestep 失效
  - `lambda_scale` 不合适

### 问题 4: 归一化可能抵消校正
- **症状**：即使 CG 求解正确，归一化后方向可能不对
- **可能原因**：
  - LML 归一化的是 `inversed_pred`（校正后）
  - HCG 归一化的是 `c_t * corrected_noise`（缩放+校正后）
  - 两者的归一化对象不同

### 问题 5: 缺少 EMA 平滑
- **症状**：噪声预测不稳定，校正抖动
- **可能原因**：
  - LML 使用 `prev_noise` 进行 EMA 平滑
  - HCG 没有利用历史信息

## 控制变量法排查方案

### 测试 1: 禁用 c_t 缩放，使用固定 lambda（最接近 LML）
```bash
--use_spectral_radius False \
--use_adaptive_lambda False \
--lambda_base 0.004 \
--use_normalization True \
--use_cg_warm_start False
```
**目标**：模拟 LML 的固定 lambda，禁用 c_t 缩放，看基本框架是否工作

### 测试 2: 启用 c_t 缩放，禁用自适应 lambda
```bash
--use_spectral_radius True \
--use_adaptive_lambda False \
--lambda_base 0.004
```
**目标**：看是否是 c_t 缩放的问题

### 测试 3: 禁用 c_t 缩放，启用自适应 lambda
```bash
--use_spectral_radius False \
--use_adaptive_lambda True \
--lambda_scale 0.3
```
**目标**：看是否是自适应 lambda 的问题

### 测试 4: 收紧 CG 容差，增加最大迭代次数
```bash
--cg_tol 1e-5 \
--cg_max_iter 20 \
--use_cg_warm_start False
```
**目标**：看是否是 CG 求解不准确的问题

### 测试 5: 禁用 CG warm start
```bash
--no_use_cg_warm_start
```
**目标**：看是否是 warm start 导致的问题

### 测试 6: 添加 EMA 平滑（类似 LML）
```bash
--use_ema_smoothing \
--ema_kappa 1e-8
```
**目标**：看是否缺少平滑导致的问题

### 测试 7: 禁用归一化
```bash
--no_use_normalization
```
**目标**：看是否是归一化的问题

### 测试 8: 跳过 Lanczos，使用固定特征值（快速测试）
```bash
--unuse_lanczos_estimation \
--fixed_alpha 1.0 \
--fixed_beta 0.1
```
**目标**：排除特征值估计的问题，快速测试其他部分

### 测试 9: 组合测试 - 最接近 LML 的配置
```bash
--use_spectral_radius False \
--use_adaptive_lambda False \
--lambda_base 0.004 \
--use_ema_smoothing \
--ema_kappa 1e-8 \
--use_normalization True \
--use_cg_warm_start False \
--cg_tol 1e-5 \
--cg_max_iter 20
```
**目标**：最接近 LML 的配置，看是否还有问题

## 建议的排查顺序

1. **先测试 8**：跳过 Lanczos，使用固定特征值，快速验证其他部分是否正常
2. **再测试 1**：禁用 c_t 和自适应 lambda，使用固定 lambda=0.004，最接近 LML
3. **然后测试 5 和 7**：分别禁用 warm start 和归一化，看是否有改善
4. **接着测试 6**：添加 EMA 平滑，看是否有改善
5. **测试 4**：收紧 CG 容差，增加迭代次数
6. **测试 2 和 3**：分别启用 c_t 和自适应 lambda，找出问题来源
7. **最后测试 9**：组合所有优化，看最终效果

## 新增的控制变量说明

### `--use_cg_warm_start` / `--no_use_cg_warm_start`
- **作用**：控制是否使用上一次的 CG 解作为初始猜测
- **默认**：True（启用）
- **影响**：如果上一次解不准确，可能导致收敛到错误解

### `--use_normalization` / `--no_use_normalization`
- **作用**：控制是否在校正后归一化噪声幅度
- **默认**：True（启用）
- **影响**：归一化可能改变校正后的方向

### `--use_ema_smoothing`
- **作用**：启用 EMA 平滑，类似 LML 方法
- **默认**：False（禁用）
- **影响**：利用历史信息，减少噪声抖动

### `--ema_kappa`
- **作用**：EMA 平滑因子，与 LML 的 kappa 相同
- **默认**：1e-8
- **影响**：控制历史信息的权重

### `--unuse_lanczos_estimation`
- **作用**：跳过 Lanczos 特征值估计，使用固定值
- **默认**：False（禁用）
- **影响**：快速测试，排除特征值估计的问题

### `--fixed_alpha` / `--fixed_beta`
- **作用**：当 `unuse_lanczos_estimation=True` 时使用的固定特征值
- **默认**：alpha=1.0, beta=0.1
- **影响**：用于快速测试，避免特征值估计的开销
