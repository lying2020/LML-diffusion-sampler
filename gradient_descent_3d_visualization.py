import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class OptimizationVisualizer3D:
    """Enhanced 3D Optimization Algorithm Visualization Class"""
    
    def __init__(self, A, b, x0, bounds=(-4, 4), function_type='quadratic'):
        """
        Initialize 3D optimization visualizer
        
        Args:
            A: Hessian matrix of quadratic function
            b: linear term coefficients
            x0: initial point
            bounds: visualization range
            function_type: 'quadratic', 'rosenbrock', 'rastrigin', 'ackley'
        """
        self.A = A
        self.b = b
        self.x0 = x0
        self.bounds = bounds
        self.function_type = function_type
        
        # Define objective function and gradient based on type
        if function_type == 'quadratic':
            self.f = lambda x: 0.5 * x.T @ A @ x - b.T @ x
            self.grad_f = lambda x: A @ x - b
        elif function_type == 'rosenbrock':
            self.f = lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
            self.grad_f = lambda x: np.array([
                -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
                200 * (x[1] - x[0]**2)
            ])
        elif function_type == 'rastrigin':
            self.f = lambda x: 20 + x[0]**2 + x[1]**2 - 10 * (np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))
            self.grad_f = lambda x: np.array([
                2*x[0] + 20*np.pi*np.sin(2*np.pi*x[0]),
                2*x[1] + 20*np.pi*np.sin(2*np.pi*x[1])
            ])
        elif function_type == 'ackley':
            self.f = lambda x: -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + np.e + 20
            self.grad_f = lambda x: np.array([
                4*x[0]*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2)))/np.sqrt(0.5*(x[0]**2 + x[1]**2)) + 2*np.pi*np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1])))*np.sin(2*np.pi*x[0]),
                4*x[1]*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2)))/np.sqrt(0.5*(x[0]**2 + x[1]**2)) + 2*np.pi*np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1])))*np.sin(2*np.pi*x[1])
            ])
        
        # Store algorithm results
        self.results = {}
    
    def steepest_descent(self, max_iter=20, tol=1e-6):
        """Steepest Descent Method"""
        path = [self.x0.copy()]
        gradients = [self.grad_f(self.x0)]
        function_values = [self.f(self.x0)]
        
        start_time = time.time()
        
        for k in range(max_iter):
            xk = path[-1]
            gk = self.grad_f(xk)
            
            # Check convergence
            if np.linalg.norm(gk) < tol:
                break
                
            # Search direction is negative gradient
            dk = -gk
            
            # Line search for optimal step size
            if self.function_type == 'quadratic':
                # Exact line search for quadratic function
                alpha_k = (gk.T @ gk) / (dk.T @ self.A @ dk)
            else:
                # Backtracking line search for non-quadratic functions
                alpha_k = self._backtracking_line_search(xk, dk, gk)
            
            # Calculate next point
            x_next = xk + alpha_k * dk
            g_next = self.grad_f(x_next)
            
            # Record path and gradients
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
    
    def conjugate_gradient(self, max_iter=20, tol=1e-6):
        """Conjugate Gradient Method"""
        path = [self.x0.copy()]
        gradients = [self.grad_f(self.x0)]
        function_values = [self.f(self.x0)]
        directions = []
        
        start_time = time.time()
        
        for k in range(max_iter):
            xk = path[-1]
            gk = self.grad_f(xk)
            
            # Check convergence
            if np.linalg.norm(gk) < tol:
                break
            
            # Calculate search direction
            if k == 0:
                # First iteration uses negative gradient direction
                dk = -gk
            else:
                # Calculate β_k (Polak-Ribiere formula)
                beta_k = (gk.T @ gk) / (gradients[-2].T @ gradients[-2])
                dk = -gk + beta_k * directions[-1]
            
            # Line search for optimal step size
            if self.function_type == 'quadratic':
                # Exact line search for quadratic function
                alpha_k = -(gk.T @ dk) / (dk.T @ self.A @ dk)
            else:
                # Backtracking line search for non-quadratic functions
                alpha_k = self._backtracking_line_search(xk, dk, gk)
            
            # Calculate next point
            x_next = xk + alpha_k * dk
            g_next = self.grad_f(x_next)
            
            # Record path and gradients
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
    
    def _backtracking_line_search(self, x, d, g, alpha=1.0, c1=1e-4, rho=0.5):
        """Backtracking line search for non-quadratic functions"""
        f_x = self.f(x)
        g_d = g.T @ d
        
        while self.f(x + alpha * d) > f_x + c1 * alpha * g_d:
            alpha *= rho
            if alpha < 1e-10:  # Prevent infinite loop
                break
        return alpha
    
    def create_3d_surface(self, resolution=50):
        """Create 3D surface data"""
        x = np.linspace(self.bounds[0], self.bounds[1], resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.f(np.array([X[i, j], Y[i, j]]))
        
        return X, Y, Z
    
    def plot_3d_comparison(self, save_path=None):
        """Plot 3D comparison of both algorithms with 2D projections"""
        # Create 3D surface
        X, Y, Z = self.create_3d_surface()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(24, 8))
        
        # Color settings
        colors = {'steepest_descent': 'blue', 'conjugate_gradient': 'red'}
        labels = {'steepest_descent': 'Steepest Descent', 'conjugate_gradient': 'Conjugate Gradient'}
        
        # Plot 1: Steepest Descent
        ax1 = fig.add_subplot(131, projection='3d')
        self._plot_3d_algorithm_with_2d_projection(ax1, 'steepest_descent', X, Y, Z, colors, labels)
        ax1.set_title('Steepest Descent Method\n(3D + 2D Projection)', fontsize=14, fontweight='bold')
        
        # Plot 2: Conjugate Gradient
        ax2 = fig.add_subplot(132, projection='3d')
        self._plot_3d_algorithm_with_2d_projection(ax2, 'conjugate_gradient', X, Y, Z, colors, labels)
        ax2.set_title('Conjugate Gradient Method\n(3D + 2D Projection)', fontsize=14, fontweight='bold')
        
        # Plot 3: Comparison Overlay
        ax3 = fig.add_subplot(133, projection='3d')
        self._plot_3d_comparison_overlay_with_2d(ax3, X, Y, Z, colors, labels)
        ax3.set_title('Algorithm Comparison\n(3D + 2D Projections)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_3d_algorithm_with_2d_projection(self, ax, algorithm, X, Y, Z, colors, labels):
        """Plot single algorithm in 3D with 2D projection showing zigzag"""
        if algorithm not in self.results:
            return
        
        result = self.results[algorithm]
        path = result['path']
        gradients = result['gradients']
        
        # Plot 3D surface
        surf = ax.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis', linewidth=0, antialiased=True)
        
        # Plot contour lines on the surface
        ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap='viridis', alpha=0.5)
        
        # Plot 3D iteration path
        path_3d = np.column_stack([path, [self.f(p) for p in path]])
        ax.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2], 
                'o-', color=colors[algorithm], linewidth=3, markersize=8, 
                label=labels[algorithm])
        
        # Plot 2D projection on the bottom plane (showing zigzag pattern)
        ax.plot(path[:, 0], path[:, 1], Z.min(), 
                'o-', color=colors[algorithm], linewidth=2, markersize=6, 
                alpha=0.7, linestyle='--', label=f'{labels[algorithm]} (2D Projection)')
        
        # Annotate points
        for i, (x_point, z_val) in enumerate(zip(path, [self.f(p) for p in path])):
            ax.text(x_point[0], x_point[1], z_val + 0.5, f'$x_{i}$', 
                   fontsize=10, color='red')
        
        # Plot gradient vectors as simple lines (avoiding Arrow3D issues)
        scale_factor = 0.2
        for i, (x_point, grad) in enumerate(zip(path, gradients)):
            if i < len(path) - 1:
                z_val = self.f(x_point)
                # Simple line instead of arrow
                ax.plot([x_point[0], x_point[0] + grad[0] * scale_factor],
                       [x_point[1], x_point[1] + grad[1] * scale_factor],
                       [z_val, z_val + 0.1], 'g-', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f(x_1, x_2)$')
        ax.legend()
    
    def _plot_3d_comparison_overlay_with_2d(self, ax, X, Y, Z, colors, labels):
        """Plot 3D comparison overlay with 2D projections"""
        # Plot 3D surface
        surf = ax.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis', linewidth=0, antialiased=True)
        
        # Plot contour lines
        ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap='viridis', alpha=0.5)
        
        # Plot both algorithm paths with 2D projections
        for algorithm in ['steepest_descent', 'conjugate_gradient']:
            if algorithm in self.results:
                result = self.results[algorithm]
                path = result['path']
                path_3d = np.column_stack([path, [self.f(p) for p in path]])
                
                # 3D path
                ax.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2], 
                       'o-', color=colors[algorithm], linewidth=3, markersize=8, 
                       label=labels[algorithm])
                
                # 2D projection
                ax.plot(path[:, 0], path[:, 1], Z.min(), 
                       'o-', color=colors[algorithm], linewidth=2, markersize=6, 
                       alpha=0.7, linestyle='--', label=f'{labels[algorithm]} (2D)')
        
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f(x_1, x_2)$')
        ax.legend()
    
    def plot_zigzag_analysis(self):
        """Plot detailed zigzag analysis for steepest descent"""
        if 'steepest_descent' not in self.results:
            print("Need to run steepest descent first")
            return
        
        result = self.results['steepest_descent']
        path = result['path']
        gradients = result['gradients']
        
        # Calculate zigzag metrics
        zigzag_angles = []
        step_lengths = []
        
        for i in range(len(path) - 2):
            # Calculate angle between consecutive steps
            v1 = path[i+1] - path[i]
            v2 = path[i+2] - path[i+1]
            if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)  # Avoid numerical errors
                angle = np.arccos(cos_angle) * 180 / np.pi
                zigzag_angles.append(angle)
                step_lengths.append(np.linalg.norm(v1))
        
        # Create visualization
        fig = plt.figure(figsize=(20, 6))
        
        # 3D path with zigzag analysis
        ax1 = fig.add_subplot(131, projection='3d')
        X, Y, Z = self.create_3d_surface(resolution=30)
        surf = ax1.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis')
        
        # Plot path
        path_3d = np.column_stack([path, [self.f(p) for p in path]])
        ax1.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2], 
                'o-', color='blue', linewidth=3, markersize=8)
        
        # 2D projection showing zigzag
        ax1.plot(path[:, 0], path[:, 1], Z.min(), 
                'o-', color='blue', linewidth=2, markersize=6, 
                alpha=0.7, linestyle='--')
        
        ax1.set_title('Steepest Descent: Zigzag Pattern Analysis')
        ax1.set_xlabel('$x_1$')
        ax1.set_ylabel('$x_2$')
        ax1.set_zlabel('$f(x_1, x_2)$')
        
        # Zigzag angles
        ax2 = fig.add_subplot(132)
        if zigzag_angles:
            ax2.plot(range(1, len(zigzag_angles)+1), zigzag_angles, 'o-', color='red', linewidth=2)
            ax2.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='Perfect Orthogonality (90°)')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Angle between consecutive steps (degrees)')
            ax2.set_title('Zigzag Angle Analysis')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Step lengths
        ax3 = fig.add_subplot(133)
        if step_lengths:
            ax3.plot(range(1, len(step_lengths)+1), step_lengths, 'o-', color='purple', linewidth=2)
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Step Length')
            ax3.set_title('Step Length Analysis')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_convergence_3d(self):
        """Plot 3D convergence visualization"""
        if len(self.results) < 2:
            print("Need to run both algorithms to plot convergence comparison")
            return
        
        fig = plt.figure(figsize=(18, 6))
        
        # Function value convergence
        ax1 = fig.add_subplot(131)
        for algorithm, result in self.results.items():
            plt.semilogy(result['function_values'], 'o-', 
                        label=f'{algorithm} (iterations: {result["iterations"]})')
        plt.xlabel('Iterations')
        plt.ylabel('Function Value (log scale)')
        plt.title('Function Value Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gradient norm convergence
        ax2 = fig.add_subplot(132)
        for algorithm, result in self.results.items():
            grad_norms = [np.linalg.norm(g) for g in result['gradients']]
            plt.semilogy(grad_norms, 'o-', 
                        label=f'{algorithm} (final: {result["final_grad_norm"]:.2e})')
        plt.xlabel('Iterations')
        plt.ylabel('Gradient Norm (log scale)')
        plt.title('Gradient Norm Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3D path comparison with 2D projections
        ax3 = fig.add_subplot(133, projection='3d')
        X, Y, Z = self.create_3d_surface(resolution=30)
        surf = ax3.plot_surface(X, Y, Z, alpha=0.2, cmap='viridis')
        
        colors = {'steepest_descent': 'blue', 'conjugate_gradient': 'red'}
        for algorithm, result in self.results.items():
            path = result['path']
            path_3d = np.column_stack([path, [self.f(p) for p in path]])
            ax3.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2], 
                    'o-', color=colors[algorithm], linewidth=2, markersize=6, 
                    label=algorithm)
            # 2D projection
            ax3.plot(path[:, 0], path[:, 1], Z.min(), 
                   'o-', color=colors[algorithm], linewidth=1, markersize=4, 
                   alpha=0.7, linestyle='--')
        
        ax3.set_xlabel('$x_1$')
        ax3.set_ylabel('$x_2$')
        ax3.set_zlabel('$f(x_1, x_2)$')
        ax3.set_title('3D Path Comparison with 2D Projections')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
    
    def print_comparison_summary(self):
        """Print algorithm comparison summary"""
        print("\n" + "="*80)
        print(f"3D Optimization Algorithm Performance Comparison - {self.function_type.upper()} Function")
        print("="*80)
        
        for algorithm, result in self.results.items():
            print(f"\n{algorithm.upper()}:")
            print(f"  Iterations: {result['iterations']}")
            print(f"  Computation time: {result['time']:.4f} seconds")
            print(f"  Final function value: {result['function_values'][-1]:.6f}")
            print(f"  Final gradient norm: {result['final_grad_norm']:.2e}")
            print(f"  Initial function value: {result['function_values'][0]:.6f}")
            print(f"  Function improvement: {result['function_values'][0] - result['function_values'][-1]:.6f}")
        
        # Performance comparison
        if len(self.results) == 2:
            sd_result = self.results['steepest_descent']
            cg_result = self.results['conjugate_gradient']
            
            print(f"\nPerformance Comparison:")
            print(f"  Iteration ratio: CG/SD = {cg_result['iterations']}/{sd_result['iterations']} = {cg_result['iterations']/sd_result['iterations']:.2f}")
            print(f"  Time ratio: CG/SD = {cg_result['time']/sd_result['time']:.2f}")
            print(f"  Convergence precision ratio: CG/SD = {cg_result['final_grad_norm']/sd_result['final_grad_norm']:.2e}")


def main():
    """Main function with multiple function types"""
    # Test different function types
    function_types = ['quadratic', 'rosenbrock', 'rastrigin', 'ackley']
    
    for func_type in function_types:
        print(f"\n{'='*100}")
        print(f"Testing {func_type.upper()} Function")
        print('='*100)
        
        if func_type == 'quadratic':
            A = np.array([[10, 3], [3, 2]])
            b = np.array([0, 0])
            x0 = np.array([-3.0, 3.0])
            bounds = (-4, 4)
        elif func_type == 'rosenbrock':
            A = None  # Not used for non-quadratic functions
            b = None
            x0 = np.array([-1.5, 2.0])
            bounds = (-2, 2)
        elif func_type == 'rastrigin':
            A = None
            b = None
            x0 = np.array([2.0, 2.0])
            bounds = (-3, 3)
        elif func_type == 'ackley':
            A = None
            b = None
            x0 = np.array([2.0, 2.0])
            bounds = (-3, 3)
        
        # Create 3D visualizer
        visualizer = OptimizationVisualizer3D(A, b, x0, bounds, func_type)
        
        print(f"Starting 3D optimization algorithms for {func_type} function...")
        
        # Run steepest descent
        print("Running steepest descent...")
        visualizer.steepest_descent(max_iter=20)
        
        # Run conjugate gradient
        print("Running conjugate gradient...")
        visualizer.conjugate_gradient(max_iter=20)
        
        # Plot 3D comparison with 2D projections
        print("Generating 3D visualization plots with 2D projections...")
        visualizer.plot_3d_comparison()
        
        # Plot zigzag analysis for steepest descent
        print("Analyzing zigzag pattern...")
        visualizer.plot_zigzag_analysis()
        
        # Plot 3D convergence
        visualizer.plot_convergence_3d()
        
        # Print comparison summary
        visualizer.print_comparison_summary()
        
        print(f"\n{func_type.upper()} function analysis complete!")
        print("You can interact with the 3D plots by:")
        print("- Dragging to rotate the 3D view")
        print("- Using mouse wheel to zoom in/out")
        print("- Right-click and drag to pan the view")
        print("- Notice the 2D projections showing zigzag patterns!")


if __name__ == "__main__":
    main()
