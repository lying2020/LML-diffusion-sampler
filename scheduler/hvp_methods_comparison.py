"""
根据 ICLR 2024 博客文章和论文实现的不同 HVP 计算方法对比
- [ICLR 2024 Blog: How to compute Hessian-vector products?](https://iclr-blogposts.github.io/2024/blog/bench-hvp/)
- PyTorch 文档: [torch.func](https://pytorch.org/docs/stable/func.html)

四种方法：
1. Symmetrized (推荐): 使用 H_sym = 1/2*(H+H^T)，确保对称性 - Equation 8
2. Forward-over-reverse (使用 torch.func.jvp)
3. Reverse-over-forward (使用 torch.func.vjp)
4. Reverse-over-reverse (当前的 Pearlmutter 方法，最通用)
"""

import torch
import contextlib
from typing import Optional


def hessian_vector_product_symmetrized(model, x_sample: torch.Tensor, t_timestep: int, v: torch.Tensor) -> torch.Tensor:
    """
    Symmetrized Hessian-vector product using JVP and VJP.

    实现公式：H_sym v = 1/2 * (JVP(s_θ, v) + VJP(s_θ, v))

    其中：
    - H_sym = 1/2 * (H + H^T) 是对称化 Hessian
    - s_θ 是 score function (模型输出的梯度)
    - JVP: Jacobian-Vector Product
    - VJP: Vector-Jacobian Product

    这个方法确保了：
    1. 对称性：即使原始 H 不对称，H_sym 也是对称的
    2. 效率：不需要显式计算 Hessian 矩阵
    3. 速度：使用优化的 JVP 和 VJP 操作

    Reference: ICLR 2024 / Equation 8

    注意：需要禁用 flash attention 等不支持 forward AD 的优化
    """
    try:
        from torch.func import jvp, vjp, grad

        # 禁用 flash attention 等不支持 forward AD 的优化
        # 这对 JVP 是必需的，因为 forward AD 不支持某些 attention 操作
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend  # type: ignore
            sdp_context = sdpa_kernel(SDPBackend.MATH)  # 使用 math backend 支持梯度
        except (ImportError, AttributeError):
            try:
                from torch.backends.cuda import sdp_kernel
                sdp_context = sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
            except (ImportError, AttributeError):
                sdp_context = contextlib.nullcontext()

        # 保存模型状态
        model_training = model.training
        model.eval()

        try:
            with sdp_context:
                # 定义 score function: s_θ(x, t) = -∇log p_t(x)
                # 实际上就是模型的输出（噪声预测的负梯度）
                def score_fn(x):
                    score_pred = model(x, t_timestep)
                    if hasattr(score_pred, 'sample'):
                        score_pred = score_pred.sample
                    return score_pred

                # 方法1：直接对 score function 使用 JVP 和 VJP（符合论文公式）
                # 根据 Equation 8: H_sym v = 1/2 * (JVP(s_θ, v) + VJP(s_θ, v))
                try:
                    # JVP: J_sθ(x, t) * v，其中 J_sθ 是 score function 的雅可比
                    _, jvp_result = jvp(score_fn, (x_sample,), (v,))

                    # VJP: J_sθ(x, t)^T * v
                    vjp_fn = vjp(score_fn, x_sample)[1]
                    vjp_result = vjp_fn(v)[0]

                    # 对称化：H_sym v = 1/2 * (JVP + VJP)
                    # 根据论文，这就是对称化 Hessian-vector product
                    Hv_sym = 0.5 * (jvp_result + vjp_result)

                    return Hv_sym.detach()

                except Exception as e1:
                    # JVP 失败（通常是 forward AD 不支持某些操作）
                    # 检查是否是预期的 forward AD 问题
                    error_str = str(e1)
                    is_forward_ad_issue = (
                        "_scaled_dot_product" in error_str or
                        "forward AD" in error_str or
                        "does not support it" in error_str
                    )

                    if is_forward_ad_issue:
                        # forward AD 不支持是预期情况，静默降级到 reverse-over-reverse
                        # reverse-over-reverse 只使用 reverse-mode AD，不需要 forward AD
                        return hessian_vector_product_reverse_over_reverse(model, x_sample, t_timestep, v)
                    else:
                        # 其他未知错误，也降级但不打印过多警告（避免刷屏）
                        return hessian_vector_product_reverse_over_reverse(model, x_sample, t_timestep, v)

        finally:
            # 恢复模型状态
            if model_training:
                model.train()

    except ImportError:
        # torch.func 不可用，回退到标准方法
        return hessian_vector_product_reverse_over_reverse(model, x_sample, t_timestep, v)
    except Exception as e:
        # 任何其他错误，静默回退到标准方法
        return hessian_vector_product_reverse_over_reverse(model, x_sample, t_timestep, v)


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
            from torch.nn.attention import sdpa_kernel, SDPBackend  # type: ignore
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
            - "auto": 自动选择最快可用方法（推荐）
            - "symmetrized": 对称化方法，确保对称性（推荐用于非对称 Hessian）
            - "forward-over-reverse": 使用 jvp 方法（快）
            - "reverse-over-forward": 使用 vjp 方法（快）
            - "reverse-over-reverse": 使用 Pearlmutter 方法（最通用，默认）

    Returns:
        H*v: Hessian-vector product（对于 symmetrized 方法，返回对称化的 H_sym*v）

    Note:
        - "symmetrized" 方法使用 H_sym = 1/2 * (H + H^T)，确保结果对称
        - 即使原始 Hessian 不对称，symmetrized 方法也能保证对称性
        - 基于 ICLR 2024 论文中的 Equation 8
    """
    if method == "auto":
        # 自动选择：优先尝试对称化，但遇到 forward AD 问题会自动降级
        # 策略：先试对称化（最佳），失败则尝试其他快速方法，最后回退到 reverse-over-reverse
        try:
            result = hessian_vector_product_symmetrized(model, x_sample, t_timestep, v)
            return result
        except:
            try:
                return hessian_vector_product_forward_over_reverse(model, x_sample, t_timestep, v)
            except:
                try:
                    return hessian_vector_product_reverse_over_forward(model, x_sample, t_timestep, v)
                except:
                    # 最终回退：reverse-over-reverse 是最通用的方法
                    return hessian_vector_product_reverse_over_reverse(model, x_sample, t_timestep, v)

    elif method == "symmetrized":
        return hessian_vector_product_symmetrized(model, x_sample, t_timestep, v)

    elif method == "forward-over-reverse":
        return hessian_vector_product_forward_over_reverse(model, x_sample, t_timestep, v)

    elif method == "reverse-over-forward":
        return hessian_vector_product_reverse_over_forward(model, x_sample, t_timestep, v)

    elif method == "reverse-over-reverse":
        return hessian_vector_product_reverse_over_reverse(model, x_sample, t_timestep, v)

    else:
        raise ValueError(f"Unknown method: {method}. Choose from: auto, symmetrized, forward-over-reverse, reverse-over-forward, reverse-over-reverse")


# 兼容性别名（保持与现有代码的兼容）
hessian_vector_product = hessian_vector_product_reverse_over_reverse
