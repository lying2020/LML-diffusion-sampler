# cifar_zigzag_demo.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
import torch
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# -----------------------
# 1. 载入 CIFAR-10 (只取 train 的一部分以加速)
# -----------------------
def load_cifar10_samples(n_samples=5000):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root='./data/cifar10/cifar-10-batches-py', train=True, download=True, transform=transform)
    # 取前 n_samples 个样本
    imgs = []
    for i in range(min(n_samples, len(dataset))):
        img, _ = dataset[i]  # img: Tensor CxHxW in [0,1]
        # 将图像转换成灰度并展平（也可以用彩色每通道）
        gray = 0.2989*img[0].numpy() + 0.5870*img[1].numpy() + 0.1140*img[2].numpy()
        imgs.append(gray.ravel())
    X = np.stack(imgs, axis=0).astype(np.float64)
    # 标准化（零均值）
    X -= X.mean(axis=0, keepdims=True)
    return X

# -----------------------
# 2. PCA -> 2D
# -----------------------
def compute_pca2(X):
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    return X2, pca

# -----------------------
# 3. 估计高斯 (mu, Sigma)
# -----------------------
def estimate_gaussian(X2):
    mu = X2.mean(axis=0)
    cov = np.cov(X2.T)  # 2x2
    return mu, cov

# -----------------------
# 4. 模拟 Langevin SDE（欧拉—马尔科夫离散）
#    dx = grad_log_p(x) * dt + sqrt(2*dt) * Normal(0, I)
#    grad_log_p = -Sigma^{-1} (x - mu)  (高斯情况)
#    Preconditioned (LML-like): use H_inv @ grad
# -----------------------
def simulate_langevin(mu, cov, x0=None, steps=500, dt=0.01, precondition=False, seed=0):
    np.random.seed(seed)
    Sigma = cov
    Sigma_inv = np.linalg.inv(Sigma)
    # Hessian of log p is -Sigma_inv, so H_inv = (-Sigma_inv)^{-1} = -Sigma
    # LML uses (H + lambda I)^{-1} @ grad; for simplicity we use H_inv = -Sigma (no damping).
    H_inv = -Sigma  # note the negative sign; we'll apply it carefully below

    if x0 is None:
        x = mu + np.array([2.0, 1.0])  # 初始位置（在 PCA 空间）
    else:
        x = np.array(x0, dtype=float)
    traj = [x.copy()]
    for _ in range(steps):
        grad_logp = - Sigma_inv.dot(x - mu)  # ∇ log p
        if precondition:
            # LML-like: apply H_inv @ grad_logp
            # since H_inv = -Sigma, H_inv @ grad_logp = -Sigma @ (-Sigma_inv (x-mu)) = x - mu
            drift = H_inv.dot(grad_logp)  # equals (x-mu)
        else:
            drift = grad_logp
        noise = np.sqrt(2*dt)*np.random.randn(2)
        x = x + drift * dt + noise
        traj.append(x.copy())
    return np.array(traj)

# -----------------------
# 5. 可视化
# -----------------------
def plot_results(X2, mu, cov, traj_std, traj_precond):
    # 密度等高线（高斯）
    x = np.linspace(X2[:,0].min()-1, X2[:,0].max()+1, 200)
    y = np.linspace(X2[:,1].min()-1, X2[:,1].max()+1, 200)
    Xg, Yg = np.meshgrid(x,y)
    pos = np.stack([Xg, Yg], axis=-1)
    inv_cov = np.linalg.inv(cov)
    Z = np.exp(-0.5 * ((pos - mu) @ inv_cov * (pos - mu)).sum(axis=-1))
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.contour(Xg, Yg, Z, levels=20)
    plt.scatter(X2[:,0], X2[:,1], s=2, alpha=0.2)
    plt.plot(traj_std[:,0], traj_std[:,1], 'r-', label='Standard Langevin')
    plt.scatter(traj_std[0,0], traj_std[0,1], c='r', marker='o')
    plt.title('Standard Langevin in PCA2')
    plt.legend()
    plt.axis('equal')

    plt.subplot(1,2,2)
    plt.contour(Xg, Yg, Z, levels=20)
    plt.scatter(X2[:,0], X2[:,1], s=2, alpha=0.2)
    plt.plot(traj_precond[:,0], traj_precond[:,1], 'b-', label='Preconditioned (LML-like)')
    plt.scatter(traj_precond[0,0], traj_precond[0,1], c='b', marker='o')
    plt.title('Preconditioned Langevin (H^{-1} applied)')
    plt.legend()
    plt.axis('equal')

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(results_dir, 'zigzag_cifar10.png'))

# -----------------------
# main
# -----------------------
if __name__ == "__main__":
    print("加载 CIFAR-10（可能会花一点时间）...")
    X = load_cifar10_samples(n_samples=5000)  # 可改为更小，例如 2000
    print("做 PCA -> 2D")
    X2, pca = compute_pca2(X)
    mu, cov = estimate_gaussian(X2)
    print("mu:", mu)
    print("cov:\n", cov)
    # 初始点选择：离均值较远一点
    x0 = mu + np.array([2.5, 0.6])
    traj_std = simulate_langevin(mu, cov, x0=x0, steps=800, dt=0.005, precondition=False, seed=42)
    traj_pre = simulate_langevin(mu, cov, x0=x0, steps=800, dt=0.005, precondition=True, seed=42)
    plot_results(X2, mu, cov, traj_std, traj_pre)
