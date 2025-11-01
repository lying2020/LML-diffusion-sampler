# HCG 中间变量和参数保存指南

## 当前已保存的变量

1. **`eigenvalue_cache`**: 特征值缓存
   - 格式: `{timestep: (timestep, (alpha_t, beta_t))}`
   - 用途: 避免重复计算，提高效率

2. **`prev_cg_solution`**: 前一步的CG解
   - 格式: `torch.Tensor`
   - 用途: 作为热启动初始猜测，加速CG收敛

3. **`prev_noise`**: 前一步的噪声预测
   - 格式: `torch.Tensor`
   - 用途: 用于DPM-Solver的多步更新

## 建议补充保存的中间变量

### 1. 诊断统计信息（用于分析和调试）

```python
# 在 scheduler 的 __init__ 中添加
self.hcg_stats = {
    'lambda_t_history': [],           # lambda_t 的值序列
    'kappa_current_history': [],      # 实际条件数序列
    'kappa_target_history': [],       # 目标条件数序列
    'c_t_history': [],               # 谱半径缩放因子序列
    'cg_iterations_history': [],     # CG迭代次数序列
    'cg_residuals_history': [],      # CG最终残差序列
    'alpha_t_history': [],           # 最大特征值序列
    'beta_t_history': [],            # 最小特征值序列
    'hvp_call_count': 0,             # HVP调用总次数
    'eigenvalue_estimates_count': 0, # 特征值估计次数
}
```

### 2. 性能统计信息

```python
self.performance_stats = {
    'lanczos_time': [],              # Lanczos估计耗时
    'cg_time': [],                   # CG求解耗时
    'hvp_time': [],                  # HVP计算耗时
    'adaptive_damping_time': [],     # 自适应阻尼计算耗时
    'total_hcg_time': [],            # 总HCG耗时
}
```

### 3. 数值稳定性指标

```python
self.numerical_stability = {
    'cg_convergence_rate': [],       # CG收敛率
    'condition_number_achieved': [], # 达到的条件数
    'spectral_radius_used': [],      # 使用的谱半径
}
```

## 根据PDF文档可能需要的额外变量

基于常见的HCG实现和理论要求，可能需要：

1. **Hessian对角元素近似** (可选)
   - 用于快速初始化或预处理
   - `H_diag_approx`: 对角元素近似

2. **CG搜索方向历史** (用于分析)
   - 可以保存最近几次的搜索方向

3. **HVP计算结果缓存** (高级优化)
   - 缓存最近几次HVP结果，用于近似

4. **时间步相关的状态**
   - 当前时间步对应的sigma_t, alpha_t等
   - 用于验证数值稳定性

## 实现建议

### 方案1: 完整统计模式（用于研究）
- 保存所有中间变量
- 占用较多内存，但可用于深入分析

### 方案2: 精简统计模式（默认）
- 只保存关键统计信息
- 内存占用小，适合生产环境

### 方案3: 可配置统计模式（推荐）
- 通过参数控制保存哪些统计
- 平衡内存和诊断需求

## 变量保存的优先级

**高优先级（必须）**：
- lambda_t, alpha_t, beta_t (已有缓存)
- CG迭代次数（用于性能分析）
- 条件数（用于验证算法正确性）

**中优先级（推荐）**：
- c_t 谱半径缩放因子
- CG残差历史
- 时间统计

**低优先级（可选）**：
- 完整的CG搜索方向
- HVP结果缓存
- 详细的数值稳定性指标
