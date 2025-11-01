# HCG Profiling 优化说明

## 优化内容

### 1. 移除重复的 Profiling
- **`regularized_hessian_vector_product`**: 移除外层 profiling，避免与 `hessian_vector_product` 内部的细粒度 profiling 重复统计
- **原因**: 该函数只是轻量级包装（添加正则化项），实际时间消耗都在 `hessian_vector_product` 内部

### 2. 添加 HVP 调用统计
- **`cg_hvp_call`**: 在 CG 循环中单独统计每次 HVP 调用
  - 包括 warm start 的初始 HVP 调用
  - 包括每次 CG 迭代的 HVP 调用
- **`lanczos_hvp_call`**: 在 Lanczos 算法中统计每次 HVP 调用
  - 包括每个随机向量的初始 HVP 调用
  - 包括每次 Lanczos 迭代的 HVP 调用

### 3. 添加细粒度 Profiling
- **`adaptive_damping_compute`**: lambda_t 的计算时间
- **`spectral_scaling_compute`**: c_t 的计算时间
- **`spectral_scaling_apply`**: c_t * corrected_noise 的缩放操作时间
- **`hcg_normalization`**: 归一化操作时间

## Profiling 层次结构

```
hcg_correct (整体函数，由 @profile_function 装饰器自动添加)
├── lanczos_eigenvalue_estimation (特征值估计)
│   ├── lanczos_hvp_call × (num_vectors × (1 + k))
│   │   └── hessian_vector_product (内部已有细粒度 profiling)
│   │       ├── model_forward_pass
│   │       ├── first_gradient
│   │       └── second_gradient (或 finite_difference_hvp)
│   └── ... (Lanczos 其他操作)
├── adaptive_damping_compute (lambda_t 计算)
├── spectral_scaling_compute (c_t 计算)
├── conjugate_gradient_solve (CG 求解)
│   ├── cg_hvp_call × (num_iterations + warm_start)
│   │   └── regularized_hessian_vector_product
│   │       └── hessian_vector_product (内部已有细粒度 profiling)
│   │           ├── model_forward_pass
│   │           ├── first_gradient
│   │           └── second_gradient
│   └── ... (CG 其他操作)
├── spectral_scaling_apply (c_t 缩放)
└── hcg_normalization (归一化)
```

## 性能分析指标

### 关键统计项

1. **总体性能**:
   - `hcg_correct`: 总 HCG 校正时间

2. **特征值估计**:
   - `lanczos_eigenvalue_estimation`: 总时间
   - `lanczos_hvp_call`: 调用次数 ≈ num_vectors × (1 + k)

3. **CG 求解**:
   - `conjugate_gradient_solve`: 总时间
   - `cg_hvp_call`: 调用次数 = num_iterations + (1 if warm_start else 0)

4. **其他操作**:
   - `adaptive_damping_compute`: lambda_t 计算时间
   - `spectral_scaling_compute`: c_t 计算时间
   - `spectral_scaling_apply`: 缩放操作时间
   - `hcg_normalization`: 归一化时间

5. **HVP 细粒度统计** (来自 `hessian_vector_product` 内部):
   - `model_forward_pass`: 模型前向传播
   - `first_gradient`: 一阶梯度计算
   - `second_gradient`: 二阶梯度计算
   - `finite_difference_hvp`: 有限差分回退 (如果启用)

## 使用方式

```python
from utils.profiling import get_profiler

# 启用 profiler
profiler = get_profiler()
profiler.enabled = True

# ... 运行 HCG ...

# 查看报告（按总时间排序，显示前10项）
profiler.print_report(sort_by='total_time', top_n=10, logger=logger)

# 保存报告到 JSON
profiler.save_report('profiling_report.json')
```

## 优化收益

1. **性能分析更清晰**: 可以区分 Lanczos 和 CG 中的 HVP 调用次数和耗时
2. **移除重复统计**: 避免 `regularized_hessian_vector_product` 的重复 profiling
3. **细粒度分析**: 每个关键操作都有独立的时间统计
4. **更好的调试**: 可以快速定位性能瓶颈

## 注意事项

1. **Profiling 开销**: profiling 本身有开销，如果不需要可以禁用 `profiler.enabled = False`
2. **CUDA 同步**: CUDA profiling 会强制同步，可能影响性能测量准确性
3. **嵌套 profiling**: 已确保不嵌套相同名称的 profiling，避免重复统计
