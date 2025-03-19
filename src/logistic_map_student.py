import numpy as np
import matplotlib.pyplot as plt
import os

# 创建结果目录
os.makedirs("results", exist_ok=True)

def iterate_logistic(r, x0, n_iter):
    """
    Logistic映射迭代函数
    参数:
        r (float): 增长率参数
        x0 (float): 初始值 (0 ≤ x0 ≤ 1)
        n_iter (int): 迭代次数
    返回:
        x (ndarray): 迭代后的序列
    """
    x = np.zeros(n_iter)
    x[0] = x0
    for i in range(1, n_iter):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

def plot_time_series():
    """
    绘制四个r值的时间序列子图
    返回:
        fig (Figure): matplotlib图像对象
    """
    r_values = [2, 3.2, 3.45, 3.6]
    n_iter = 60
    x0 = 0.5

    # 创建2x2子图
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("不同r值的时间序列分析", fontsize=14)

    for idx, r in enumerate(r_values):
        # 计算迭代序列
        x = iterate_logistic(r, x0, n_iter)
        
        # 确定子图位置
        ax = axs[idx//2, idx%2]
        ax.plot(x, "b-", linewidth=1)
        ax.set_xlabel("迭代次数", fontsize=9)
        ax.set_ylabel("x值", fontsize=9)
        ax.set_title(f"r = {r}", fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_feigenbaum():
    """
    绘制费根鲍姆图（分岔图）
    返回:
        fig (Figure): matplotlib图像对象
    """
    r_min = 2.6
    r_max = 4.0
    n_r = 1401  # 对应步长0.001
    n_iter = 250
    n_discard = 100

    # 生成r值数组
    r_values = np.linspace(r_min, r_max, n_r)
    x = np.zeros(n_iter)
    r_plot = []
    x_plot = []

    # 遍历所有r值
    for r in r_values:
        # 初始化并迭代
        x[0] = 0.5
        for i in range(1, n_iter):
            x[i] = r * x[i-1] * (1 - x[i-1])
        
        # 记录稳定后的数据
        r_plot.extend([r] * (n_iter - n_discard))
        x_plot.extend(x[n_discard:])

    # 绘制分岔图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(r_plot, x_plot, ",k", markersize=0.1, alpha=0.3)
    ax.set_xlabel("r值", fontsize=10)
    ax.set_ylabel("x稳定值", fontsize=10)
    ax.set_title("Logistic映射的费根鲍姆图", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    return fig

if __name__ == "__main__":
    # 任务1：时间序列图
    time_series_fig = plot_time_series()
    time_series_fig.savefig("results/time_series.png", dpi=300, bbox_inches="tight")
    plt.close(time_series_fig)

    # 任务2：费根鲍姆图
    feigenbaum_fig = plot_feigenbaum()
    feigenbaum_fig.savefig("results/feigenbaum.png", dpi=300, bbox_inches="tight")
    plt.close(feigenbaum_fig)

    print("实验图像已保存至results目录")
