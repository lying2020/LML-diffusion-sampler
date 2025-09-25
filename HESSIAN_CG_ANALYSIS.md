# Hessian Free 和 CG（共轭梯度算法）在 LML-Diffusion-Sampler 中的使用分析

## 1. 整体架构概览

LML (Levenberg-Marquardt Langevin) 扩散采样器使用了三种不同的 Hessian 计算方法：

```
LML Correction Methods:
├── Original LML (简化近似)
├── Explicit Hessian (显式 Hessian 计算)
└── Hessian-Free (使用 CG + HVP)
```

## 2. 核心算法流程

### 2.1 LML 修正的基本思想

LML 修正的目标是求解以下优化问题：
```
minimize: ||ε - ε_pred||² + λ||ε||²
subject to: ε 满足扩散模型的约束
```

其中：
- `ε` 是噪声预测
- `ε_pred` 是模型预测的噪声
- `λ` 是正则化参数
- 需要求解 Hessian 矩阵的逆：`H⁻¹g`

### 2.2 三种 Hessian 计算方法

#### 方法1: Original LML (简化近似)
```python
def lm_correct_original(noise_pred, noise_pred_ema, lamb):
    # 使用简化的 Hessian 近似
    norm_squared = (noise_pred * noise_pred).sum(dim=(1, 2, 3))
    inner_product = torch.sum(noise_pred * noise_pred_ema, dim=(1, 2, 3))

    # 近似 Hessian 逆：H⁻¹ ≈ (I + λI)⁻¹
    part1 = noise_pred
    part2 = noise_pred_ema * inner_product / (lamb + norm_squared_ema)

    corrected_noise = part1 - part2
    return corrected_noise
```

#### 方法2: Explicit Hessian (显式计算)
```python
def lm_correct_explicit_hessian(noise_pred, noise_pred_ema, lamb, model, x, t, device):
    # 1. 计算梯度
    x_grad = x.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        score_pred = model(x_grad, t)
        log_prob = -0.5 * torch.sum(score_pred ** 2, dim=(1, 2, 3))
        log_prob = log_prob.sum()

    grad = torch.autograd.grad(log_prob, x_grad, create_graph=True)[0]

    # 2. 使用有限差分近似 Hessian
    hessian_diag = torch.zeros_like(x)
    for i in range(channels):
        for j in range(height):
            for k in range(width):
                # 扰动单个元素
                x_pert = x.clone()
                x_pert[:, i, j, k] += eps

                # 计算扰动后的梯度
                grad_pert = compute_gradient(x_pert, model, t)
                hessian_diag[:, i, j, k] = (grad_pert[:, i, j, k] - grad[:, i, j, k]) / eps

    # 3. 应用对角 Hessian 近似
    hessian_inv_diag = 1.0 / (hessian_diag + lamb)
    corrected_noise = noise_pred * hessian_inv_diag
    return corrected_noise
```

#### 方法3: Hessian-Free (使用 CG + HVP)
```python
def lm_correct_hessian_free(noise_pred, noise_pred_ema, lamb, model, x, t, device):
    def hessian_vector_product(v):
        """使用 Pearlmutter 方法计算 Hv"""
        v_grad = v.clone().detach().requires_grad_(True)
        x_grad = x.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            score_pred = model(x_grad, t)
            log_prob = -0.5 * torch.sum(score_pred ** 2, dim=(1, 2, 3))
            log_prob = log_prob.sum()

        # 计算梯度
        grad = torch.autograd.grad(log_prob, x_grad, create_graph=True)[0]

        # 计算 Hv = ∇(∇f · v)
        grad_dot_v = torch.sum(grad * v_grad)
        hv = torch.autograd.grad(grad_dot_v, x_grad, retain_graph=True)[0]
        return hv

    def conjugate_gradient_solve(b, max_iter=20, tol=1e-4):
        """使用共轭梯度求解 Hx = b"""
        x_cg = torch.zeros_like(b)
        r = b.clone()  # 残差
        p = r.clone()  # 搜索方向

        r_norm_sq = torch.sum(r ** 2)
        r_norm_0 = torch.sqrt(r_norm_sq)

        for i in range(max_iter):
            Hp = hessian_vector_product(p)  # 计算 Hp
            p_Hp = torch.sum(p * Hp)

            if p_Hp <= 0:  # 检查正定性
                break

            # CG 步骤
            alpha = r_norm_sq / p_Hp
            x_cg = x_cg + alpha * p
            r = r - alpha * Hp

            r_norm_sq_new = torch.sum(r ** 2)
            r_norm = torch.sqrt(r_norm_sq_new)

            if r_norm < tol * r_norm_0:  # 收敛检查
                break

            beta = r_norm_sq_new / r_norm_sq
            p = r + beta * p
            r_norm_sq = r_norm_sq_new

        return x_cg

    # 求解 H⁻¹ * noise_pred
    corrected_noise = conjugate_gradient_solve(noise_pred)
    corrected_noise = corrected_noise / (1.0 + lamb)  # 正则化
    return corrected_noise
```

## 3. 关键技术细节

### 3.1 Pearlmutter 方法 (Hessian-Vector Product)

Pearlmutter 方法是一种高效计算 Hessian-向量乘积的技术：

```python
def hessian_vector_product(v):
    # 1. 计算一阶梯度
    grad = ∇f(x)

    # 2. 计算梯度与向量的内积
    grad_dot_v = grad · v

    # 3. 计算二阶梯度
    hv = ∇(grad_dot_v) = ∇(∇f · v)

    return hv
```

**优势**：
- 不需要显式存储 Hessian 矩阵
- 内存复杂度：O(n) 而不是 O(n²)
- 计算复杂度：O(n) 而不是 O(n²)

### 3.2 共轭梯度算法 (Conjugate Gradient)

CG 算法用于求解线性方程组 `Hx = b`：

```python
def conjugate_gradient(b, max_iter, tol):
    x = 0
    r = b  # 初始残差
    p = r  # 初始搜索方向

    for i in range(max_iter):
        Hp = hessian_vector_product(p)
        alpha = (r·r) / (p·Hp)  # 步长
        x = x + alpha * p       # 更新解
        r = r - alpha * Hp      # 更新残差

        if ||r|| < tol:         # 收敛检查
            break

        beta = (r_new·r_new) / (r_old·r_old)
        p = r + beta * p        # 更新搜索方向
```

**优势**：
- 对于正定矩阵，最多 n 步收敛
- 每次迭代只需要一次 Hessian-向量乘积
- 内存效率高

### 3.3 自适应正则化

在高级实现中，还包含了自适应正则化：

```python
def adaptive_damping_function(condition_number, method='adaptive'):
    if method == 'adaptive':
        if condition_number > 1000:
            return 0.1
        elif condition_number > 100:
            return 0.01
        else:
            return 0.001
    else:
        return 0.001
```

## 4. 算法比较

| 方法 | 计算复杂度 | 内存复杂度 | 精度 | 适用场景 |
|------|------------|------------|------|----------|
| Original LML | O(n) | O(n) | 低 | 快速近似 |
| Explicit Hessian | O(n²) | O(n²) | 中 | 中等规模 |
| Hessian-Free | O(kn) | O(n) | 高 | 大规模问题 |

其中 k 是 CG 迭代次数，通常 k << n。

## 5. 实际应用流程

### 5.1 在扩散采样中的集成

```python
class DPMSolverMultistepHessianFreeScheduler:
    def step(self, model_output, timestep, sample, **kwargs):
        # 1. 标准 DPM 步骤
        prev_sample = self.dpm_solver_step(model_output, timestep, sample)

        # 2. LML 修正
        if self.lm:
            corrected_output = lm_correct_advanced(
                prev_noise=self.prev_noise,
                noise_pred=model_output,
                lamb=self.lamb,
                kappa=self.kappa,
                hessian_method=self.hessian_method,  # 'original', 'explicit', 'hessian_free'
                model=self.model,
                x=sample,
                t=timestep,
                device=self.device
            )
            return corrected_output

        return prev_sample
```

### 5.2 参数调优

```python
# 不同场景的参数设置
configs = {
    'fast': {'hessian_method': 'original', 'lamb': 0.001},
    'balanced': {'hessian_method': 'explicit', 'lamb': 0.01},
    'accurate': {'hessian_method': 'hessian_free', 'lamb': 0.1, 'max_iter': 20}
}
```

## 6. 性能优化策略

### 6.1 内存优化
- 使用梯度检查点减少内存使用
- 批处理 Hessian-向量乘积计算
- 及时释放中间变量

### 6.2 计算优化
- 预计算常用的梯度
- 使用混合精度计算
- 并行化 CG 迭代

### 6.3 收敛优化
- 自适应步长调整
- 预处理技术
- 早停策略

## 7. 总结

Hessian-Free 和 CG 算法在 LML-diffusion-sampler 中的使用体现了以下特点：

1. **理论严谨性**：基于 Levenberg-Marquardt 优化理论
2. **计算效率**：使用 Pearlmutter 方法避免显式 Hessian 存储
3. **数值稳定性**：通过正则化和自适应参数调整
4. **可扩展性**：支持不同规模和精度的应用场景

这种设计使得 LML 修正能够在保持计算效率的同时，显著提升扩散模型的采样质量。
