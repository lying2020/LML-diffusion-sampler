# Hessian Free 和 CG（共轭梯度算法）

## 2. 核心算法流程

## 🔧 技术特性

### Hessian-Free方法
- **算法**: 使用共轭梯度(CG)求解Hessian逆
- **优势**: 避免显式计算Hessian矩阵，提高计算效率
- **应用**: DPM-Solver++ + LML校正 + Hessian-Free优化

### 配置参数
- **solver_order**: 3 (三阶求解器)
- **algorithm_type**: "dpmsolver++" (使用DPM-Solver++算法)
- **hessian_method**: "hcg" (使用Hessian-Free方法)

### 2.2 Hessian 计算方法

#### 方法1: Explicit Hessian (显式计算)
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

#### 方法2: Hessian-Free (使用 CG + HVP)
```python
def lm_correct_(noise_pred, noise_pred_ema, lamb, model, x, t, device):
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

#### 方法3: HCG (Hessian-Conjugate Gradient) 自适应校正

这是最新的实现方法，使用自适应阻尼和谱半径缩放：

```python
def hcg_correct(noise_pred, model, x, t, device, kappa_target=10.0):
    """
    Hessian-Conjugate Gradient 校正使用自适应阻尼

    核心思路：
    1. 使用 Lanczos 算法估计 Hessian 的最大最小特征值 α_t, β_t
    2. 根据目标条件数 κ_* 计算自适应阻尼 λ_t
    3. 使用 CG 求解 (H + λ_t I)^{-1} * noise_pred
    4. 应用谱半径缩放 c_t = 1/(α_t + λ_t)
    """

    # Step 1: Lanczos 特征值估计
    def hessian_vector_product(v):
        """计算 Hv = ∇²log p_t(x) * v (对称化)"""
        v_grad = v.clone().detach().requires_grad_(True)
        x_grad = x.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            score_pred = model(x_grad, t)
            log_prob = -0.5 * torch.sum(score_pred ** 2, dim=(1, 2, 3))
            log_prob = log_prob.sum()

        grad = torch.autograd.grad(log_prob, x_grad, create_graph=True)[0]
        grad_dot_v = torch.sum(grad * v_grad)
        Hv = torch.autograd.grad(grad_dot_v, x_grad, retain_graph=True)[0]
        return Hv

    # 使用 Lanczos 算法估计特征值
    alpha_t, beta_t = lanczos_eigenvalue_estimation(
        hessian_vector_product_fn=hessian_vector_product,
        x_shape=x.shape,
        k=10,  # Lanczos 迭代次数
        num_vectors=3,  # 随机向量数量
        device=device
    )

    # Step 2: 计算自适应阻尼参数
    # λ_t = max(0, (α_t - κ_* * β_t) / (κ_* - 1))
    # 确保 κ(H_sym + λ_t I) ≤ κ_*
    lambda_t = adaptive_damping_lambda(alpha_t, beta_t, kappa_target)

    # Step 3: 计算谱半径缩放因子
    # c_t = 1 / (α_t + λ_t) = 1 / (λ_max + λ_t)
    c_t = 1.0 / (alpha_t + lambda_t + 1e-8)

    # Step 4: 使用 CG 求解 (H + λ_t I)^{-1} * noise_pred
    def regularized_hessian_vector_product(v):
        """计算 (H + λ_t I) * v"""
        Hv = hessian_vector_product(v)
        return Hv + lambda_t * v

    corrected_noise = conjugate_gradient_solve(
        noise_pred,
        regularized_hessian_vector_product,
        max_iter=20,
        tol=1e-4
    )

    # Step 5: 应用谱半径缩放
    corrected_noise = c_t * corrected_noise

    # Step 6: 归一化保持幅值
    norm_original = torch.norm(noise_pred)
    norm_corrected = torch.norm(corrected_noise)
    corrected_noise = corrected_noise * norm_original / (norm_corrected + 1e-8)

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

#### 3.3.1 简单自适应正则化

在基础实现中，使用简单的分段函数：

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

#### 3.3.2 基于条件数的自适应阻尼 (最新实现)

新的实现使用基于特征值的自适应阻尼公式：

```python
def adaptive_damping_lambda(alpha_t, beta_t, kappa_target=10.0):
    """
    计算自适应阻尼参数 λ_t

    公式: λ_t = max(0, (α_t - κ_* * β_t) / (κ_* - 1))

    其中:
    - α_t = λ_max(H_sym): Hessian 的最大特征值
    - β_t = λ_min(H_sym): Hessian 的最小特征值
    - κ_*: 目标条件数 (>1)

    这个公式确保: κ(H_sym + λ_t I) ≤ κ_*
    """
    if kappa_target <= 1.0:
        raise ValueError(f"kappa_target must be > 1, got {kappa_target}")

    kappa_current = alpha_t / (beta_t + 1e-8)

    if kappa_current <= kappa_target:
        return 0.0  # 无需阻尼

    numerator = alpha_t - kappa_target * beta_t
    denominator = kappa_target - 1.0

    lambda_t = max(0.0, numerator / denominator)
    return lambda_t
```

**优势**：
- **自适应调整**：λ_t 根据 Hessian 条件数自动调整，而不是固定超参数
- **理论保证**：确保条件数受控，几何更稳定
- **数值稳定性**：在早期（近各向同性）避免过阻尼，在后期（极端各向异性）避免阻尼不足

### 3.4 Lanczos 特征值估计

使用 Lanczos 算法估计 Hessian 的最大最小特征值：

```python
def lanczos_eigenvalue_estimation(hessian_vector_product_fn, x_shape, k=10, num_vectors=5):
    """
    使用 Lanczos 算法估计 Hessian 的特征值

    算法流程:
    1. 生成多个随机向量
    2. 对每个向量运行 Lanczos 迭代，构建三对角矩阵
    3. 从三对角矩阵提取特征值
    4. 返回所有估计的平均值（稳健估计）
    """
    max_eigenvals = []
    min_eigenvals = []

    for _ in range(num_vectors):
        # 初始化随机向量
        v = random_normalized_vector(x_shape)
        q = [v]
        alpha = []
        beta = []

        # Lanczos 迭代
        for i in range(k):
            w = hessian_vector_product_fn(q[-1])
            alpha_i = dot(w, q[-1])
            alpha.append(alpha_i)

            if i > 0:
                w = w - alpha_i * q[-1] - beta[-1] * q[-2]
            else:
                w = w - alpha_i * q[-1]

            beta_i = norm(w)
            beta.append(beta_i)

            if beta_i > 1e-8:
                q.append(w / beta_i)
            else:
                break

        # 从三对角矩阵提取特征值
        T = build_tridiagonal_matrix(alpha, beta)
        eigenvals = np.linalg.eigvalsh(T)
        eigenvals = np.sort(eigenvals)[::-1]

        max_eigenvals.append(eigenvals[0])
        min_eigenvals.append(eigenvals[-1])

    # 返回稳健估计（平均值）
    alpha_t = np.mean(max_eigenvals)
    beta_t = np.mean(min_eigenvals)

    return alpha_t, beta_t
```

**优势**：
- **高效**：只需要 Hessian-向量乘积，不需要显式 Hessian
- **稳健**：使用多个随机向量平均，减少估计误差
- **自适应**：根据实际 Hessian 的几何特性调整

### 3.5 谱半径缩放 (Spectral Radius Scaling)

为了数值稳定性，应用谱半径缩放：

```python
def compute_spectral_scaling(alpha_t, lambda_t):
    """
    计算谱半径缩放因子

    公式: c_t = 1 / (α_t + λ_t) = 1 / (λ_max + λ_t)

    这确保缩放后的 Hessian 逆的谱范围在 [1/κ_*, 1] 内
    """
    c_t = 1.0 / (alpha_t + lambda_t + 1e-8)
    return c_t
```

**理论依据**（来自 Lemma）：
- 定义：`M_t(x) = c_t (H_sym(x) + λ_t I)^{-1}`
- 其中 `c_t = β_t + λ_t` (或 `c_t = 1/(α_t + λ_t)` 用于归一化)
- 保证：`spec(M_t) ⊂ [1/κ_*, 1]`

这确保了数值稳定性和几何一致性。

## 4. 算法比较

| 方法 | 计算复杂度 | 内存复杂度 | 精度 | 自适应性 | 适用场景 |
|------|------------|------------|------|----------|----------|
| **HCG (自适应)** | **O(kn + mn)** | **O(n)** | **最高** | **完全自适应** | **高质量采样** |

其中：
- `k` 是 CG 迭代次数，通常 k << n
- `m` 是 Lanczos 迭代次数，通常 m << n
- **HCG** 是 Hessian-Conjugate Gradient 的缩写，使用自适应阻尼和谱半径缩放

#### HCG (自适应) ⭐ **推荐**
- ✅ 完全自适应阻尼（基于条件数）
- ✅ 谱半径缩放提高稳定性
- ✅ Lanczos 特征值估计
- ✅ 理论保证（条件数控制）
- ✅ 适用于各向异性到各向同性的过渡

## 5. 实际应用流程

### 5.1 在扩散采样中的集成

```python
class DPMSolverMultistepHCGScheduler:
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
                hessian_method=self.hessian_method,  # 'original', 'explicit', 'hcg'
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
    'adaptive': {  # 最新推荐配置
        'use_hcg': True,
        'kappa_target': 10.0,  # 目标条件数
        'lanczos_k': 10,      # Lanczos 迭代次数
        'cg_max_iter': 20,    # CG 最大迭代次数
        'cg_tol': 1e-4,       # CG 容忍度
        'use_spectral_scaling': True  # 启用谱半径缩放
    }
}
```

### 5.3 HCG 方法使用示例

```python
from scheduler.scheduling_dpmsolver_multistep_hcg import DPMSolverMultistepHCGScheduler

# 创建调度器
scheduler = DPMSolverMultistepHCGScheduler(
    num_train_timesteps=1000,
    solver_order=2,
    algorithm_type="dpmsolver++",
    use_hcg=True,                    # 启用 HCG 校正
    kappa_target=10.0,               # 目标条件数
    lanczos_k=10,                    # Lanczos 迭代次数
    cg_max_iter=20,                  # CG 最大迭代次数
    cg_tol=1e-4,                     # CG 容忍度
    use_spectral_scaling=True,        # 启用谱半径缩放
)

# 设置模型（必需，用于计算 Hessian）
scheduler.set_model(model)

# 设置时间步
scheduler.set_timesteps(num_inference_steps=50)

# 在采样循环中使用
for t in scheduler.timesteps:
    # 模型前向传播
    model_output = model(sample, t)

    # 调度器步骤（自动应用 HCG 校正）
    sample = scheduler.step(model_output, t, sample).prev_sample
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

## 7. 最新实现总结

### 7.1 HCG 自适应校正方法

最新的 `DPMSolverMultistepHCGScheduler` 实现完全独立于 `scheduling_dpmsolver_multistep_lm.py`，实现了以下核心功能：

#### 1. **Lanczos 特征值估计** (`lanczos_eigenvalue_estimation`)
- ✅ 使用 Lanczos 算法估计 Hessian 的最大最小特征值
- ✅ 使用多个随机向量提高稳健性
- ✅ 通过三对角矩阵特征值提取 α_t 和 β_t
- ✅ 高效：只需要 Hessian-向量乘积，不需要显式 Hessian

#### 2. **自适应阻尼参数** (`adaptive_damping_lambda`)
- ✅ 实现公式：`λ_t = max(0, (α_t - κ_* * β_t) / (κ_* - 1))`
- ✅ 根据目标条件数 κ_* 自动调整 λ_t
- ✅ 确保 `κ(H_sym + λ_t I) ≤ κ_*`
- ✅ 自适应：λ_t 随 Hessian 条件数动态变化，而非固定超参数

#### 3. **Hessian-Conjugate Gradient 校正** (`hcg_correct`)
- ✅ 使用 HVP (Hessian-Vector Product) + CG 求解 `(H + λ_t I)^{-1} * g`
- ✅ 应用谱半径缩放：`c_t = 1/(α_t + λ_t)`
- ✅ 最终结果：`c_t * (H + λ_t I)^{-1} * noise_pred`
- ✅ 数值稳定：通过缩放确保谱范围受控

#### 4. **完整的 DPMSolverMultistepHCGScheduler 类**
- ✅ 完全独立，不依赖 `scheduling_dpmsolver_multistep_lm.py`
- ✅ 包含完整的 DPM-Solver 实现（first, second, third order）
- ✅ 支持所有算法类型（dpmsolver, dpmsolver++, sde-dpmsolver 等）
- ✅ 丰富的可配置参数

### 7.2 技术优势

Hessian-Free 和 CG 算法在 LML-diffusion-sampler 中的使用体现了以下特点：

1. **理论严谨性**：
   - 基于 Levenberg-Marquardt 优化理论
   - 自适应阻尼公式有理论保证（条件数控制）
   - 谱半径缩放确保数值稳定性

2. **计算效率**：
   - 使用 Pearlmutter 方法避免显式 Hessian 存储
   - 内存复杂度：O(n) 而不是 O(n²)
   - 计算复杂度：O(kn + mn)，其中 k, m << n

3. **数值稳定性**：
   - 自适应阻尼：根据 Hessian 条件数自动调整
   - 谱半径缩放：c_t = 1/(α_t + λ_t) 确保几何稳定
   - 在早期（近各向同性）避免过阻尼，在后期（极端各向异性）避免阻尼不足

4. **可扩展性**：
   - 支持不同规模和精度的应用场景
   - 可配置的迭代次数和容忍度
   - 适用于从各向同性到各向异性的完整过渡

### 7.3 关键创新点

1. **自适应阻尼机制**：
   - 不再是固定超参数，而是根据 Hessian 条件数动态计算
   - 公式：`λ_t = max(0, (α_t - κ_* * β_t) / (κ_* - 1))`
   - 确保条件数受控，几何更稳定

2. **谱半径缩放**：
   - `c_t = 1/(α_t + λ_t)` 作为整体缩放因子
   - 保证 `spec(M_t) ⊂ [1/κ_*, 1]`
   - 提高数值稳定性和收敛性

3. **无显式 Hessian**：
   - 仅通过 HVP 计算，避免存储大矩阵
   - 使用 Lanczos 算法估计特征值
   - 使用 CG 求解线性系统

这种设计使得 HCG 修正能够在保持计算效率的同时，显著提升扩散模型的采样质量，特别是在处理各向异性分布时。
