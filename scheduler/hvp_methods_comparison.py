"""
根据 ICLR 2024 博客文章实现的不同 HVP 计算方法对比
- [ICLR 2024 Blog: How to compute Hessian-vector products?](https://iclr-blogposts.github.io/2024/blog/bench-hvp/)
- PyTorch 文档: [torch.func](https://pytorch.org/docs/stable/func.html)

三种方法：
1. Forward-over-reverse (使用 torch.func.jvp)
2. Reverse-over-forward (使用 torch.func.vjp)
3. Reverse-over-reverse (当前的 Pearlmutter 方法)
"""

import torch
import contextlib
from typing import Optional


def hessian_vector_product_forward_over_reverse(model, x_sample: torch.Tensor, t_timestep: int, v: torch.Tensor) -> torch.Tensor:
    """
    Forward-over-reverse method for computing HVP.

    这是最快的方法之一。使用 JAX/PyTorch 的 jvp (Jacobian-vector product) 函数。
    对于 f: R^d -> R，计算 H*v = J_grad(v)，其中 J_grad 是梯度的雅可比矩阵。

    算法：
    1. 计算梯度 g = ∇f(x)
    2. 使用 jvp 计算 J_g(v) = ∇(∇f(x) · v) = H*v

    时间复杂度：O(gradient computation)
    内存复杂度：O(gradient computation)
    """
    try:
        from torch.func import jvp, grad

        # 定义函数：f(x) -> loss
        def loss_fn(x):
            score_pred = model(x, t_timestep)
            if hasattr(score_pred, 'sample'):
                score_pred = score_pred.sample
            # log_prob = -0.5 * ||score_pred||^2
            log_prob = -0.5 * torch.sum(score_pred ** 2, dim=(1, 2, 3))
            return log_prob.sum()

        # 计算梯度函数 g(x) = ∇f(x)
        grad_fn = grad(loss_fn)

        # 使用 jvp 计算 H*v = J_g(v)
        # jvp 计算：在方向 v 上的梯度变化率
        _, Hv = jvp(grad_fn, (x_sample,), (v,))

        return Hv

    except ImportError:
        # Fallback: PyTorch < 2.0 可能没有 torch.func
        raise ImportError("torch.func is required for forward-over-reverse method. Use PyTorch >= 2.0")
    except Exception as e:
        # Fallback to reverse-over-reverse if jvp fails
        print(f"Warning: forward-over-reverse failed ({e}), falling back to reverse-over-reverse")
        return hessian_vector_product_reverse_over_reverse(model, x_sample, t_timestep, v)


def hessian_vector_product_reverse_over_forward(model, x_sample: torch.Tensor, t_timestep: int, v: torch.Tensor) -> torch.Tensor:
    """
    Reverse-over-forward method for computing HVP.

    这也是最快的方法之一。使用 vjp (vector-Jacobian product) 函数。

    算法：
    1. 定义辅助函数 g_v(x) = <∇f(x), v>
    2. 计算梯度 ∇g_v(x) = H*v

    时间复杂度：O(gradient computation)
    内存复杂度：O(gradient computation)
    """
    try:
        from torch.func import vjp, grad

        # 定义函数：f(x) -> loss
        def loss_fn(x):
            score_pred = model(x, t_timestep)
            if hasattr(score_pred, 'sample'):
                score_pred = score_pred.sample
            log_prob = -0.5 * torch.sum(score_pred ** 2, dim=(1, 2, 3))
            return log_prob.sum()

        # 计算梯度函数 g(x) = ∇f(x)
        grad_fn = grad(loss_fn)

        # 定义辅助函数：g_v(x) = <∇f(x), v>
        def grad_dot_v_fn(x):
            g = grad_fn(x)
            return torch.sum(g * v)

        # 计算 ∇g_v(x) = H*v
        Hv_fn = grad(grad_dot_v_fn)
        Hv = Hv_fn(x_sample)

        return Hv

    except ImportError:
        raise ImportError("torch.func is required for reverse-over-forward method. Use PyTorch >= 2.0")
    except Exception as e:
        # Fallback to reverse-over-reverse if vjp fails
        print(f"Warning: reverse-over-forward failed ({e}), falling back to reverse-over-reverse")
        return hessian_vector_product_reverse_over_reverse(model, x_sample, t_timestep, v)


def hessian_vector_product_reverse_over_reverse(model, x_sample: torch.Tensor, t_timestep: int, v: torch.Tensor) -> torch.Tensor:
    """
    Reverse-over-reverse method (Pearlmutter's method).

    这是当前代码使用的方法，最通用但稍慢。

    算法：
    1. 计算 g = ∇f(x)
    2. 计算 <g, v>
    3. 计算 ∇<g, v> = H*v

    时间复杂度：~2x gradient computation
    内存复杂度：~2x gradient computation (需要存储两次反向传播的中间结果)
    """
    # 确保 v 需要梯度
    v_grad = v.clone().detach().requires_grad_(True)

    # 创建需要梯度的 x
    x_grad = x_sample.clone().detach().requires_grad_(True)

    # 保存模型状态
    model_training = model.training
    model.eval()

    try:
        # 禁用 flash attention 等不支持二阶导数的优化
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            sdp_context = sdpa_kernel(SDPBackend.MATH)
        except (ImportError, AttributeError):
            try:
                from torch.backends.cuda import sdp_kernel
                sdp_context = sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
            except (ImportError, AttributeError):
                sdp_context = contextlib.nullcontext()

        with sdp_context:
            with torch.enable_grad():
                # Forward pass
                score_pred = model(x_grad, t_timestep)
                if hasattr(score_pred, 'sample'):
                    score_pred = score_pred.sample

                # log_prob = -0.5 * ||score_pred||^2
                log_prob = -0.5 * torch.sum(score_pred ** 2, dim=(1, 2, 3))
                log_prob = log_prob.sum()

                # 第一次反向传播：计算梯度 g = ∇log_prob
                grad_outputs = torch.autograd.grad(
                    outputs=log_prob,
                    inputs=x_grad,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=False,
                    retain_graph=True
                )

                if len(grad_outputs) == 0 or grad_outputs[0] is None:
                    raise RuntimeError("Failed to compute gradient.")

                grad = grad_outputs[0]

                # 计算内积 <g, v>
                grad_dot_v = torch.sum(grad * v_grad)

                # 第二次反向传播：计算 H*v = ∇<g, v>
                Hv_outputs = torch.autograd.grad(
                    outputs=grad_dot_v,
                    inputs=x_grad,
                    retain_graph=False,
                    only_inputs=True,
                    allow_unused=False
                )

                if len(Hv_outputs) == 0 or Hv_outputs[0] is None:
                    raise RuntimeError("Failed to compute Hessian-vector product.")

                Hv = Hv_outputs[0]

    except RuntimeError as e:
        raise e
    finally:
        # 恢复模型状态
        model.train(model_training)

    return Hv.detach()


def hessian_vector_product_adaptive(
    model,
    x_sample: torch.Tensor,
    t_timestep: int,
    v: torch.Tensor,
    method: str = "auto"
) -> torch.Tensor:
    """
    自适应选择 HVP 计算方法。

    Args:
        model: 神经网络模型
        x_sample: 输入样本
        t_timestep: 时间步
        v: 向量
        method: 计算方法，可选：
            - "auto": 自动选择最快可用方法
            - "forward-over-reverse": 使用 jvp 方法
            - "reverse-over-forward": 使用 vjp 方法
            - "reverse-over-reverse": 使用 Pearlmutter 方法（最通用）

    Returns:
        H*v: Hessian-vector product
    """
    if method == "auto":
        # 尝试最快的方法，失败则降级
        try:
            return hessian_vector_product_forward_over_reverse(model, x_sample, t_timestep, v)
        except:
            try:
                return hessian_vector_product_reverse_over_forward(model, x_sample, t_timestep, v)
            except:
                return hessian_vector_product_reverse_over_reverse(model, x_sample, t_timestep, v)

    elif method == "forward-over-reverse":
        return hessian_vector_product_forward_over_reverse(model, x_sample, t_timestep, v)

    elif method == "reverse-over-forward":
        return hessian_vector_product_reverse_over_forward(model, x_sample, t_timestep, v)

    elif method == "reverse-over-reverse":
        return hessian_vector_product_reverse_over_reverse(model, x_sample, t_timestep, v)

    else:
        raise ValueError(f"Unknown method: {method}. Choose from: auto, forward-over-reverse, reverse-over-forward, reverse-over-reverse")


# 兼容性别名（保持与现有代码的兼容）
hessian_vector_product = hessian_vector_product_reverse_over_reverse
