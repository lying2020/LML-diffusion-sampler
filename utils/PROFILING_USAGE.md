# Profiling 工具使用说明

类似于 MATLAB 的 `profile` 工具，用于统计每个主要接口的时间消耗。

## 功能特点

1. **自动统计函数执行时间**：使用装饰器或上下文管理器标记需要统计的函数
2. **详细的统计信息**：
   - 调用次数
   - 总时间
   - 平均时间
   - 最小/最大时间
   - 标准差
   - CUDA 时间（如果使用 GPU）
3. **生成报告**：可输出到控制台或保存为 JSON 文件

## 使用方法

### 1. 在命令行中启用 profiling

```bash
python scripts/ldm_celeba/celeba.py \
    --sampler_type hcg \
    --num_inference_steps 40 \
    --enable_profiling \
    --profile_report_file profiling_report.json
```

### 2. 在代码中使用

#### 方法 1: 使用装饰器

```python
from utils.profiling import profile_function

@profile_function("my_function")
def my_function():
    # 你的代码
    pass
```

#### 方法 2: 使用上下文管理器

```python
from utils.profiling import profile

with profile("my_code_block"):
    # 你的代码
    pass
```

#### 方法 3: 使用全局 profiler

```python
from utils.profiling import get_profiler

profiler = get_profiler()
profiler.enabled = True
profiler.clear()

# ... 运行代码 ...

# 打印报告
profiler.print_report(sort_by='total_time', top_n=20)

# 保存报告
profiler.save_report("profiling_report.json")
```

## 输出示例

```
================================================================================
Profiling Report (sorted by total_time)
================================================================================
Function                                  Calls   Total(s)     Avg(s)       Min(s)       Max(s)
--------------------------------------------------------------------------------
hcg_correct                               40      15.2345      0.3809       0.3521       0.4123
lanczos_eigenvalue_estimation             40      8.1234      0.2031       0.1892       0.2215
conjugate_gradient_solve                  400     6.7890      0.0170       0.0156       0.0189
hessian_vector_product                    2400    4.5678      0.0019       0.0015       0.0023
adaptive_damping_lambda                   40      0.0123      0.0003       0.0002       0.0004
spectral_scaling                          40      0.0045      0.0001       0.0001       0.0002
================================================================================

Total profiled time: 34.7315s
Total function calls: 2920
```

## 已添加 profiling 的函数

在 `scheduling_dpmsolver_multistep_hcg.py` 中，以下函数已经添加了 profiling：

1. `hcg_correct` - HCG 校正主函数
2. `lanczos_eigenvalue_estimation` - Lanczos 特征值估计
3. `adaptive_damping_lambda` - 自适应阻尼计算
4. `spectral_scaling` - 谱半径缩放
5. `hessian_vector_product` - Hessian-向量乘积
6. `conjugate_gradient_solve` - 共轭梯度求解

## JSON 报告格式

保存的 JSON 报告包含：

```json
{
  "timestamp": "2025-11-01T12:00:00",
  "total_time": 34.7315,
  "functions": {
    "hcg_correct": {
      "calls": 40,
      "total_time": 15.2345,
      "avg_time": 0.3809,
      "min_time": 0.3521,
      "max_time": 0.4123,
      "std_time": 0.0123,
      "cuda_total": 15.1234,
      "cuda_avg": 0.3781
    },
    ...
  },
  "sorted_functions": ["hcg_correct", "lanczos_eigenvalue_estimation", ...]
}
```

## 性能开销

Profiling 的开销非常小（通常 < 1%），主要来自：
- 时间戳记录
- 字典操作
- CUDA 事件同步（仅在使用 GPU 时）

## 注意事项

1. Profiling 默认是禁用的，需要通过 `--enable_profiling` 启用
2. CUDA 时间统计需要 GPU 可用
3. 大量函数调用时，建议使用 `top_n` 参数限制输出数量
