import numpy as np
import matplotlib.pyplot as plt

def iterate_logistic(r, x0, n):
    """
    迭代Logistic映射
    
    参数:
        r: 增长率参数
        x0: 初始值
        n: 迭代次数
        
    返回:
        x: 迭代序列数组
    """
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

def plot_time_series(r, x0, n):
    """
    绘制时间序列图
    
    参数:
        r: 增长率参数
        x0: 初始值
        n: 迭代次数
        
    返回:
        fig: matplotlib图像对象
    """
    x = iterate_logistic(r, x0, n)
    t = np.arange(n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, x, 'b-', lw=1)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('x')
    ax.set_title(f'Logistic映射时间序列 (r={r})')
    ax.grid(True)
    
    return fig

def plot_bifurcation(r_min, r_max, n_r, n_iterations, n_discard):
    """
    绘制分岔图
    
    参数:
        r_min: r的最小值
        r_max: r的最大值
        n_r: r的取值个数
        n_iterations: 每个r值的迭代次数
        n_discard: 每个r值丢弃的初始迭代点数
        
    返回:
        fig: matplotlib图像对象
    """
    r = np.linspace(r_min, r_max, n_r)
    x = np.zeros(n_iterations)
    x_plot = []
    r_plot = []
    
    for r_val in r:
        x[0] = 0.5
        for i in range(1, n_iterations):
            x[i] = r_val * x[i-1] * (1 - x[i-1])
        
        # 只保留稳定后的点
        x_plot.extend(x[n_discard:])
        r_plot.extend([r_val] * (n_iterations - n_discard))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(r_plot, x_plot, ',k', alpha=0.1, markersize=0.1)
    ax.set_xlabel('r')
    ax.set_ylabel('x')
    ax.set_title('Logistic映射分岔图')
    
    return fig
