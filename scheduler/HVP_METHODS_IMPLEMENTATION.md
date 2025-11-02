# HVP 方法实现文档

## 概述

根据 ICLR 2024 博客文章实现的不同 Hessian-Vector Product (HVP) 计算方法，用于 HCG (Hessian-Conjugate Gradient) scheduler。

**参考：**
- [ICLR 2024 Blog: How to compute Hessian-vector products?](https://iclr-blogposts.github.io/2024/blog/bench-hvp/)
- PyTorch 文档: [torch.func](https://pytorch.org/docs/stable/func.html)

## 实现的方法

### 1. Symmetrized HVP (推荐，但需 forward AD 支持)

**函数：** `hessian_vector_product_symmetrized`

**实现公式：**
```
H_sym v = 1/2 * (JVP(s_θ, v) + VJP(s_θ, v))
```

其中：
- `H_sym = 1/2 * (H + H^T)` 是对称化 Hessian
- `s_θ` 是 score function（模型输出）
- `JVP`: Jacobian-Vector Product (使用 forward-mode AD)
- `VJP`: Vector-Jacobian Product (使用 reverse-mode AD)

**特点：**
- ✅ 保证对称性：即使原始 H 不对称，H_sym 也是对称的
- ✅ 高效：不需要显式计算 Hessian 矩阵
- ✅ 速度快：使用优化的 JVP 和 VJP 操作
- ⚠️ 需要 forward AD 支持（某些模型/操作不支持）

**自动降级：** 如果 forward AD 不支持，会自动降级到 `reverse-over-reverse`

### 2. Forward-over-Reverse

**函数：** `hessian_vector_product_forward_over_reverse`

**实现：** 使用 `torch.func.jvp` 进行 forward-mode AD

**特点：**
- ✅ 速度快
- ⚠️ 需要 forward AD 支持

### 3. Reverse-over-Forward

**函数：** `hessian_vector_product_reverse_over_forward`

**实现：** 使用 `torch.func.vjp` 进行 reverse-mode AD

**特点：**
- ✅ 速度快
- ⚠️ 需要 forward AD 支持（对于某些操作）

### 4. Reverse-over-Reverse (最通用，默认)

**函数：** `hessian_vector_product_reverse_over_reverse`

**实现：** Pearlmutter 方法，两次反向传播

**特点：**
- ✅ 最稳定，适用于所有模型
- ✅ 只使用 reverse-mode AD（backpropagation）
- ✅ 不需要 forward AD 支持
- ⚠️ 相对较慢（但稳定可靠）

### 5. Adaptive (自动选择)

**函数：** `hessian_vector_product_adaptive`

**策略：**
1. 优先尝试 `symmetrized`（最佳，保证对称性）
2. 如果失败，尝试 `forward-over-reverse`
3. 如果失败，尝试 `reverse-over-forward`
4. 最终回退到 `reverse-over-reverse`（最通用）

## 使用方式

### 在 HCG Scheduler 中使用

```python
# 初始化 scheduler
scheduler = DPMSolverMultistepHCGScheduler.from_config(config)
scheduler.hvp_method = 'auto'  # 或 'symmetrized', 'reverse-over-reverse', etc.
```

### 命令行参数

```bash
--hvp_method auto              # 自动选择最佳方法（推荐）
--hvp_method symmetrized       # 对称化方法（需 forward AD）
--hvp_method reverse-over-reverse  # 最稳定的方法
--hvp_method forward-over-reverse  # 快速方法（需 forward AD）
--hvp_method reverse-over-forward  # 快速方法（需 forward AD）
```

## Forward AD 兼容性问题

### 问题描述

某些 PyTorch 操作（特别是 Flash Attention 和 Scaled Dot Product Attention）不支持 forward-mode AD，这会导致以下错误：

```
RuntimeError: Trying to use forward AD with _scaled_dot_product_efficient_attention
that does not support it...
```

### 解决方案

代码已经自动处理了这个问题：

1. **自动检测：** 识别 forward AD 不支持的错误
2. **静默降级：** 自动降级到 `reverse-over-reverse` 方法
3. **避免警告刷屏：** forward AD 不支持是预期情况，不会打印过多警告

### 禁用 Flash Attention

代码会尝试禁用 flash attention：

```python
# 优先使用 torch.nn.attention (PyTorch 2.0+)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    sdp_context = sdpa_kernel(SDPBackend.MATH)
except ImportError:
    # 回退到 torch.backends.cuda (旧版本)
    from torch.backends.cuda import sdp_kernel
    sdp_context = sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
```

## 推荐设置

### 对于大多数模型

```python
hvp_method = 'auto'  # 自动选择，会优先尝试最优方法
```

或

```python
hvp_method = 'reverse-over-reverse'  # 最稳定，保证可用
```

### 对于支持 forward AD 的模型

```python
hvp_method = 'symmetrized'  # 保证对称性，速度最快
```

## 性能对比

| 方法 | 对称性 | 速度 | 兼容性 | 推荐场景 |
|------|--------|------|--------|----------|
| symmetrized | ✅ 保证 | ⚡⚡⚡ 最快 | ⚠️ 需 forward AD | 支持 forward AD 的模型 |
| forward-over-reverse | ❌ 不一定 | ⚡⚡ 快 | ⚠️ 需 forward AD | 支持 forward AD 的模型 |
| reverse-over-forward | ❌ 不一定 | ⚡⚡ 快 | ⚠️ 需 forward AD | 支持 forward AD 的模型 |
| reverse-over-reverse | ✅ 通常 | ⚡ 中等 | ✅ 所有模型 | 默认选择，最稳定 |
| auto | ✅ 可能 | ⚡⚡ 自动 | ✅ 所有模型 | 推荐，自动选择最佳 |

## 代码结构

```
scheduler/
├── hvp_methods_comparison.py          # HVP 方法实现
├── scheduling_dpmsolver_multistep_hcg.py  # HCG Scheduler（使用 HVP 方法）
└── HVP_METHODS_IMPLEMENTATION.md      # 本文档
```

## 数学原理

### Symmetrized HVP

对于 score function `s_θ(x, t)`，其 Hessian 矩阵 `H` 可能不对称。对称化方法计算：

```
H_sym = 1/2 * (H + H^T)
H_sym v = 1/2 * (H*v + H^T*v)
```

使用 JVP 和 VJP：

```
H*v = J_sθ(x) * v        (JVP)
H^T*v = J_sθ(x)^T * v    (VJP)
```

其中 `J_sθ(x)` 是 score function 的雅可比矩阵。

### Reverse-over-Reverse (Pearlmutter)

1. 第一次反向传播：计算梯度 `g = ∇f(x)`
2. 计算内积 `<g, v>`
3. 第二次反向传播：计算 `H*v = ∇<g, v>`

## 更新日志

- **2024-01-XX**: 初始实现，包含四种 HVP 方法和自动降级机制
- **2024-01-XX**: 添加 forward AD 兼容性处理，静默降级机制
- **2024-01-XX**: 将默认方法改为 'auto'，提高兼容性

## 参考

1. [ICLR 2024 Blog: How to compute Hessian-vector products?](https://iclr-blogposts.github.io/2024/blog/bench-hvp/)
2. [PyTorch torch.func Documentation](https://pytorch.org/docs/stable/func.html)
3. Pearlmutter, B. A. (1994). Fast exact multiplication by the Hessian. Neural computation, 6(1), 147-160.
