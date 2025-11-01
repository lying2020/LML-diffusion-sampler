# HCG 方法优化指南

## 一、当前性能瓶颈分析

### 1.1 时间消耗主要来源

根据 profiling 分析，HCG 方法的主要时间消耗在：

1. **Lanczos 特征值估计** (~40-50%)
   - 需要 3 个随机向量 × 10 次 Lanczos 迭代 = 30+ 次 HVP 调用
   - 每次 HVP 需要：1 次前向传播 + 2 次反向传播

2. **Conjugate Gradient 求解** (~30-40%)
   - 最多 20 次迭代，每次需要 1 次 HVP
   - CG 收敛可能较慢，尤其在早期阶段

3. **Hessian-Vector Product (HVP)** (~20-30%)
   - 需要计算二阶导数
   - 可能与 Flash Attention 不兼容，需要 fallback

4. **特征值估计在每个时间步重复计算**
   - 没有利用时间步之间的相关性
   - 相似的 t 值应该有相似的特征值

### 1.2 与 LML 的对比

| 操作 | LML | HCG (当前) |
|------|-----|------------|
| 模型调用次数 | 0 | 30-100+ |
| 梯度计算 | 0 | 60-200+ |
| 计算复杂度 | O(n) | O(kn + mn) |
| 平均时间/步 | ~1ms | ~100-500ms |

## 二、优化策略

### 策略 1: 缓存特征值估计 (高优先级 ⭐⭐⭐)

**问题**: 每个时间步都重新计算特征值，浪费计算资源

**解决方案**:
- 缓存最近几个时间步的特征值估计
- 使用插值或指数移动平均来平滑特征值
- 只在关键时间步（如每 5-10 步）重新估计

**预期收益**: 减少 40-60% 的 Lanczos 计算时间

**实现建议**:
```python
# 在 scheduler 中添加缓存
self.eigenvalue_cache = {}  # {timestep: (alpha, beta)}
self.cache_window = 5  # 每 N 步重新估计

# 在 hcg_correct 中
def get_cached_eigenvalues(t, x_shape):
    # 查找最近的时间步
    cached_t = find_nearest_cached_timestep(t)
    if cached_t is not None and abs(t - cached_t) < cache_window:
        return self.eigenvalue_cache[cached_t]
    # 否则重新估计
    return lanczos_eigenvalue_estimation(...)
```

### 策略 2: 减少 Lanczos 迭代次数 (高优先级 ⭐⭐⭐)

**问题**: 当前 `lanczos_k=10, num_vectors=3` 导致过多 HVP 调用

**解决方案**:
- 将 `lanczos_k` 从 10 降低到 3-5
- 将 `num_vectors` 从 3 降低到 1-2
- 使用更简单的 Hutchinson 方法替代部分 Lanczos

**预期收益**: 减少 50-70% 的特征值估计时间

**实现建议**:
```python
# 快速模式：减少迭代
lanczos_k = 3  # 从 10 降到 3
num_vectors = 1  # 从 3 降到 1

# 或者使用 Hutchinson 方法的简化版本
def hutchinson_eigenvalue_estimation(...):
    # 只需要 5-10 次 HVP 调用
    # 相比 Lanczos 的 30+ 次大幅减少
```

### 策略 3: 简化特征值估计方法 (中优先级 ⭐⭐)

**问题**: Lanczos 算法复杂，实际只需要粗略估计

**解决方案**:
- **方案 A**: 使用 Hutchinson 方法（更简单，更少迭代）
- **方案 B**: 使用对角近似（最快，但精度较低）
- **方案 C**: 使用时间步相关的启发式估计

**实现建议**:
```python
def simplified_eigenvalue_estimation(hessian_vector_product_fn, x_shape, device):
    """使用 Hutchinson 方法，只需要少量 HVP 调用"""
    total_dim = int(np.prod(x_shape[1:]))
    max_eigenvals = []
    min_eigenvals = []

    # 只使用 3-5 个随机向量
    for _ in range(3):
        v = torch.randn(x_shape, device=device)
        v = v / torch.norm(v)
        Hv = hessian_vector_product_fn(v)
        # 使用 Rayleigh quotient: λ ≈ v^T H v
        eigenval = torch.sum(v * Hv)
        max_eigenvals.append(eigenval.item())
        min_eigenvals.append(-eigenval.item())  # 对称假设

    alpha_t = torch.tensor(max(max_eigenvals), device=device)
    beta_t = torch.tensor(max(min_eigenvals), device=device)
    return alpha_t, beta_t
```

### 策略 4: 优化 CG 求解 (中优先级 ⭐⭐)

**问题**: CG 可能收敛慢，特别是早期迭代

**解决方案**:
- **减少最大迭代次数**: `cg_max_iter` 从 20 降到 5-10
- **更宽松的容差**: `cg_tol` 从 1e-4 调到 1e-2
- **更好的初始猜测**: 使用上一步的解作为初始值
- **预条件化**: 使用简单的对角预条件子

**预期收益**: 减少 30-50% 的 CG 时间

**实现建议**:
```python
def conjugate_gradient_solve(b, prev_solution=None, max_iter=5, tol=1e-2):
    """使用上一步解作为初始猜测"""
    if prev_solution is not None:
        x_cg = prev_solution.clone()  # 更好的初始猜测
    else:
        x_cg = torch.zeros_like(b)

    # ... 其他 CG 代码 ...
    # 减少迭代次数
```

### 策略 5: 选择性应用 HCG (中优先级 ⭐⭐)

**问题**: 在所有时间步都使用 HCG，但早期和后期效果可能不明显

**解决方案**:
- 只在关键时间步（中等噪声水平）使用 HCG
- 使用简单的阈值判断：`if sigma_t in [0.2, 0.8]: use_hcg`
- 或者每 N 步使用一次：`if step % 5 == 0: use_hcg`

**预期收益**: 减少 50-80% 的 HCG 调用次数

**实现建议**:
```python
def should_use_hcg(t, timesteps):
    """只在关键区域使用 HCG"""
    sigma_t = self.sigma_t[t]
    # 只在中等噪声水平使用（效果最明显）
    if 0.2 <= sigma_t <= 0.8:
        return True
    return False

# 或者按间隔使用
if step_index % 3 == 0:  # 每 3 步使用一次
    use_hcg = True
```

### 策略 6: 优化 HVP 计算 (低优先级 ⭐)

**问题**: HVP 需要二阶导数，可能触发 fallback

**解决方案**:
- 默认使用有限差分近似（更快，无需二阶导数）
- 使用混合精度（FP16）加速计算
- 优化梯度图的内存管理

**实现建议**:
```python
def fast_hessian_vector_product(v, model, x, t, eps=1e-3):
    """使用有限差分近似，避免二阶导数"""
    # 只需要两次一阶梯度计算
    x_plus = (x + eps * v).requires_grad_(True)
    x_minus = (x - eps * v).requires_grad_(True)

    grad_plus = compute_gradient(model, x_plus, t)
    grad_minus = compute_gradient(model, x_minus, t)

    Hv = (grad_plus - grad_minus) / (2 * eps)
    return Hv
```

### 策略 7: 超参数优化 (高优先级 ⭐⭐⭐)

**问题**: 当前超参数设置可能不是最优

**解决方案**:
- **lambda_scale**: 当前默认 1.0，可能应该更小（0.1-0.5）
- **kappa_target**: 当前 10.0，可以尝试 20-50（允许更大条件数）
- **lanczos_k**: 降低到 3-5
- **cg_max_iter**: 降低到 5-10
- **cg_tol**: 放宽到 1e-2 或 1e-3

**建议的默认值**:
```python
kappa_target = 20.0  # 从 10.0 增加到 20.0，减少过度正则化
lambda_scale = 0.3  # 从 1.0 降低到 0.3，减少阻尼
lanczos_k = 3  # 从 10 降低到 3
num_vectors = 1  # 从 3 降低到 1
cg_max_iter = 5  # 从 20 降低到 5
cg_tol = 1e-2  # 从 1e-4 放宽到 1e-2
```

## 三、快速优化实现（最小改动）

### 3.1 立即可以做的改动

1. **降低默认迭代次数**
```python
# 在 __init__ 中
lanczos_k: int = 3,  # 从 10 降到 3
cg_max_iter: int = 5,  # 从 20 降到 5
cg_tol: float = 1e-2,  # 从 1e-4 放宽到 1e-2
```

2. **添加特征值缓存**
```python
# 在 scheduler 类中添加
self.eigenvalue_cache = {}
self.cache_interval = 5

# 在 hcg_correct 中添加缓存逻辑
```

3. **调整超参数范围**
```python
kappa_target: float = 20.0,  # 增加
lambda_scale: float = 0.3,  # 降低
```

### 3.2 预期效果

| 优化项 | 时间减少 | 质量影响 |
|--------|---------|---------|
| 降低迭代次数 | 60-70% | 轻微下降 |
| 添加缓存 | 40-50% | 无影响 |
| 超参数调优 | 0% | 可能提升 |
| **总计** | **70-85%** | **可接受** |

## 四、渐进式优化路径

### 阶段 1: 快速优化（1-2 天）
1. 降低默认迭代次数
2. 添加特征值缓存
3. 调整超参数默认值

### 阶段 2: 算法优化（3-5 天）
1. 实现 Hutchinson 特征值估计
2. 优化 CG 初始猜测
3. 实现选择性 HCG 应用

### 阶段 3: 深度优化（1-2 周）
1. 混合精度支持
2. 批量优化
3. 自定义 CUDA kernel（如果需要）

## 五、实验验证建议

### 5.1 性能测试
- 记录优化前后的时间消耗
- 对比不同参数设置的效果

### 5.2 质量测试
- 使用 FID/IS 等指标评估生成质量
- 对比优化前后与 LML 的效果

### 5.3 A/B 测试
```python
# 测试不同配置
configs = [
    {"name": "current", "lanczos_k": 10, "cg_max_iter": 20},
    {"name": "fast", "lanczos_k": 3, "cg_max_iter": 5},
    {"name": "balanced", "lanczos_k": 5, "cg_max_iter": 10},
]
```

## 六、推荐配置

### 6.1 快速模式（追求速度）
```python
use_hcg = True
kappa_target = 30.0
lambda_scale = 0.2
lanczos_k = 3
num_vectors = 1
cg_max_iter = 3
cg_tol = 1e-1
use_spectral_radius = False  # 可选：关闭谱缩放
enable_eigenvalue_cache = True
cache_interval = 5
```

### 6.2 平衡模式（速度与质量平衡）
```python
use_hcg = True
kappa_target = 20.0
lambda_scale = 0.3
lanczos_k = 5
num_vectors = 2
cg_max_iter = 5
cg_tol = 1e-2
use_spectral_radius = True
enable_eigenvalue_cache = True
cache_interval = 3
```

### 6.3 高质量模式（追求最佳效果）
```python
use_hcg = True
kappa_target = 15.0
lambda_scale = 0.5
lanczos_k = 7
num_vectors = 2
cg_max_iter = 10
cg_tol = 5e-3
use_spectral_radius = True
enable_eigenvalue_cache = True
cache_interval = 2
```

## 七、关键注意事项

1. **lambda_scale 的重要性**: 这是最重要的超参数，需要仔细调优
2. **缓存一致性**: 确保缓存不会引入误差累积
3. **数值稳定性**: 降低迭代次数时要注意数值稳定性
4. **收敛性检查**: 简化算法后需要验证收敛性

## 八、性能监控

建议添加以下监控指标：
- HCG 调用次数
- 平均 HVP 调用次数/步
- CG 平均迭代次数
- 特征值估计时间
- 总 HCG 时间/步
- lambda_t 的实际范围

这些信息有助于进一步优化。
