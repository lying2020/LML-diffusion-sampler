import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import time

class OptimizationVisualizer:
    """优化算法可视化类"""

    def __init__(self, A, b, x0, bounds=(-4, 4)):
        """
        初始化优化可视化器

        Args:
            A: 二次函数的Hessian矩阵
            b: 线性项系数
            x0: 初始点
            bounds: 可视化范围
        """
        self.A = A
        self.b = b
        self.x0 = x0
        self.bounds = bounds

        # 定义目标函数和梯度
        self.f = lambda x: 0.5 * x.T @ A @ x - b.T @ x
        self.grad_f = lambda x: A @ x - b

        # 存储算法结果
        self.results = {}

    def steepest_descent(self, max_iter=10, tol=1e-6):
        """最速下降法"""
        path = [self.x0.copy()]
        gradients = [self.grad_f(self.x0)]
        function_values = [self.f(self.x0)]

        start_time = time.time()

        for k in range(max_iter):
            xk = path[-1]
            gk = self.grad_f(xk)

            # 检查收敛性
            if np.linalg.norm(gk) < tol:
                break

            # 搜索方向为负梯度
            dk = -gk

            # 精确线搜索求最优步长
            alpha_k = (gk.T @ gk) / (dk.T @ self.A @ dk)

            # 计算下一个点
            x_next = xk + alpha_k * dk
            g_next = self.grad_f(x_next)

            # 记录路径和梯度
            path.append(x_next)
            gradients.append(g_next)
            function_values.append(self.f(x_next))

        end_time = time.time()

        self.results['steepest_descent'] = {
            'path': np.array(path),
            'gradients': gradients,
            'function_values': function_values,
            'iterations': len(path) - 1,
            'time': end_time - start_time,
            'final_grad_norm': np.linalg.norm(gradients[-1])
        }

        return self.results['steepest_descent']

    def conjugate_gradient(self, max_iter=10, tol=1e-6):
        """共轭梯度法"""
        path = [self.x0.copy()]
        gradients = [self.grad_f(self.x0)]
        function_values = [self.f(self.x0)]
        directions = []

        start_time = time.time()

        for k in range(max_iter):
            xk = path[-1]
            gk = self.grad_f(xk)

            # 检查收敛性
            if np.linalg.norm(gk) < tol:
                break

            # 计算搜索方向
            if k == 0:
                # 第一次迭代使用负梯度方向
                dk = -gk
            else:
                # 计算β_k (Polak-Ribiere公式)
                beta_k = (gk.T @ gk) / (gradients[-2].T @ gradients[-2])
                dk = -gk + beta_k * directions[-1]

            # 精确线搜索求最优步长
            alpha_k = -(gk.T @ dk) / (dk.T @ self.A @ dk)

            # 计算下一个点
            x_next = xk + alpha_k * dk
            g_next = self.grad_f(x_next)

            # 记录路径和梯度
            path.append(x_next)
            gradients.append(g_next)
            function_values.append(self.f(x_next))
            directions.append(dk)

        end_time = time.time()

        self.results['conjugate_gradient'] = {
            'path': np.array(path),
            'gradients': gradients,
            'function_values': function_values,
            'iterations': len(path) - 1,
            'time': end_time - start_time,
            'final_grad_norm': np.linalg.norm(gradients[-1])
        }

        return self.results['conjugate_gradient']

    def create_contour_plot(self):
        """创建等高线图"""
        x = np.linspace(self.bounds[0], self.bounds[1], 200)
        y = np.linspace(self.bounds[0], self.bounds[1], 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.f(np.array([X[i, j], Y[i, j]]))

        return X, Y, Z

    def plot_comparison(self, save_path=None):
        """绘制两种算法的对比图"""
        # 创建等高线
        X, Y, Z = self.create_contour_plot()

        # 创建子图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # 颜色设置
        colors = {'steepest_descent': 'blue', 'conjugate_gradient': 'red'}
        labels = {'steepest_descent': 'steepest descent', 'conjugate_gradient': 'conjugate gradient'}

        # 绘制最速下降法
        self._plot_algorithm(ax1, 'steepest_descent', X, Y, Z, colors, labels)
        ax1.set_title('steepest_descent', fontsize=14, fontweight='bold')

        # 绘制共轭梯度法
        self._plot_algorithm(ax2, 'conjugate_gradient', X, Y, Z, colors, labels)
        ax2.set_title('conjugate gradient', fontsize=14, fontweight='bold')

        # 绘制对比图
        self._plot_comparison_overlay(ax3, X, Y, Z, colors, labels)
        ax3.set_title('algorithm comparison', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def _plot_algorithm(self, ax, algorithm, X, Y, Z, colors, labels):
        """绘制单个算法的图"""
        if algorithm not in self.results:
            return

        result = self.results[algorithm]
        path = result['path']
        gradients = result['gradients']

        # 绘制等高线
        contour = ax.contour(X, Y, Z, 20, colors='gray', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)

        # 绘制迭代路径
        ax.plot(path[:, 0], path[:, 1], 'o-', color=colors[algorithm],
                linewidth=2, markersize=6, label=labels[algorithm])

        # 标注点
        for i, x_point in enumerate(path):
            ax.annotate(f'$x_{i}$', (x_point[0], x_point[1]), xytext=(5, 5),
                       textcoords='offset points', fontsize=12, color='red')

        # 绘制梯度向量
        scale_factor = 0.15
        for i, (x_point, grad) in enumerate(zip(path, gradients)):
            if i < len(path) - 1:
                arrow = FancyArrowPatch(x_point, x_point + grad * scale_factor,
                                      color='green', arrowstyle='->',
                                      mutation_scale=12, linewidth=1.5, alpha=0.8)
                ax.add_patch(arrow)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_comparison_overlay(self, ax, X, Y, Z, colors, labels):
        """绘制对比叠加图"""
        # 绘制等高线
        contour = ax.contour(X, Y, Z, 20, colors='gray', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)

        # 绘制两种算法的路径
        for algorithm in ['steepest_descent', 'conjugate_gradient']:
            if algorithm in self.results:
                result = self.results[algorithm]
                path = result['path']
                ax.plot(path[:, 0], path[:, 1], 'o-', color=colors[algorithm],
                       linewidth=2, markersize=6, label=labels[algorithm])

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_convergence(self):
        """绘制收敛性对比图"""
        if len(self.results) < 2:
            print("需要运行两种算法才能绘制收敛性对比图")
            return

        plt.figure(figsize=(12, 5))

        # 函数值收敛图
        plt.subplot(1, 2, 1)
        for algorithm, result in self.results.items():
            plt.semilogy(result['function_values'], 'o-',
                        label=f'{algorithm} (迭代次数: {result["iterations"]})')
        plt.xlabel('iterations')
        plt.ylabel('function value (log scale)')
        plt.title('function value convergence comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 梯度范数收敛图
        plt.subplot(1, 2, 2)
        for algorithm, result in self.results.items():
            grad_norms = [np.linalg.norm(g) for g in result['gradients']]
            plt.semilogy(grad_norms, 'o-',
                        label=f'{algorithm} (final gradient norm: {result["final_grad_norm"]:.2e})')
        plt.xlabel('iterations')
        plt.ylabel('gradient norm (log scale)')
        plt.title('gradient norm convergence comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def print_comparison_summary(self):
        """打印算法对比摘要"""
        print("\n" + "="*60)
        print("算法性能对比摘要")
        print("="*60)

        for algorithm, result in self.results.items():
            print(f"\n{algorithm.upper()}:")
            print(f"  迭代次数: {result['iterations']}")
            print(f"  计算时间: {result['time']:.4f} 秒")
            print(f"  最终函数值: {result['function_values'][-1]:.6f}")
            print(f"  最终梯度范数: {result['final_grad_norm']:.2e}")
            print(f"  初始函数值: {result['function_values'][0]:.6f}")
            print(f"  函数值改善: {result['function_values'][0] - result['function_values'][-1]:.6f}")

        # 性能对比
        if len(self.results) == 2:
            sd_result = self.results['steepest_descent']
            cg_result = self.results['conjugate_gradient']

            print(f"\n性能对比:")
            print(f"  迭代次数比: CG/SD = {cg_result['iterations']}/{sd_result['iterations']} = {cg_result['iterations']/sd_result['iterations']:.2f}")
            print(f"  时间比: CG/SD = {cg_result['time']/sd_result['time']:.2f}")
            print(f"  收敛精度比: CG/SD = {cg_result['final_grad_norm']/sd_result['final_grad_norm']:.2e}")


def main():
    """主函数"""
    # 定义问题参数
    A = np.array([[10, 3],
                  [3, 2]])  # 对称正定矩阵
    b = np.array([0, 0])
    x0 = np.array([-3.0, 3.0])

    # 创建可视化器
    visualizer = OptimizationVisualizer(A, b, x0)

    print("开始运行优化算法...")

    # 运行最速下降法
    print("运行最速下降法...")
    visualizer.steepest_descent(max_iter=10)

    # 运行共轭梯度法
    print("运行共轭梯度法...")
    visualizer.conjugate_gradient(max_iter=10)

    # 绘制对比图
    print("生成可视化图表...")
    visualizer.plot_comparison()

    # 绘制收敛性对比
    visualizer.plot_convergence()

    # 打印对比摘要
    visualizer.print_comparison_summary()

    # 打印详细迭代结果
    print("\n" + "="*60)
    print("详细迭代结果")
    print("="*60)

    for algorithm, result in visualizer.results.items():
        print(f"\n{algorithm.upper()} 迭代路径:")
        print("-" * 40)
        for i, x_point in enumerate(result['path']):
            f_val = result['function_values'][i]
            grad_norm = np.linalg.norm(result['gradients'][i])
            print(f"x_{i} = [{x_point[0]:.4f}, {x_point[1]:.4f}], "
                  f"f(x_{i}) = {f_val:.6f}, ||g_{i}|| = {grad_norm:.2e}")


if __name__ == "__main__":
    main()
