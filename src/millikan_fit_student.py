"""
最小二乘拟合和光电效应实验
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
   """
    加载数据文件
    
    参数:
        filename: 数据文件路径
        
    返回:
        x: 频率数据数组 (单位：Hz)
        y: 电压数据数组 (单位：V)
    """
   data = np.loadtxt(filename)
   return data[:, 0], data[:, 1]

def calculate_parameters(x, y):
    """
    计算最小二乘拟合参数
    
    参数:
        x: x坐标数组
        y: y坐标数组
        
    返回:
        m: 斜率 (单位：V/Hz)
        c: 截距 (单位：V)
        Ex: x的平均值
        Ey: y的平均值
        Exx: x^2的平均值
        Exy: xy的平均值
    """
    N = len(x)
    Ex = np.mean(x)      # x的均值
    Ey = np.mean(y)      # y的均值
    Exx = np.mean(x**2)  # x平方的均值
    Exy = np.mean(x*y)   # x*y的均值
    
    # 计算斜率和截距
    denominator = Exx - Ex**2
    if denominator == 0:
        raise ValueError("分母为零，无法计算斜率和截距")
    m = (Exy - Ex * Ey) / denominator
    c = (Exx * Ey - Ex * Exy) / denominator
    
    return m, c, Ex, Ey, Exx, Exy

def plot_data_and_fit(x, y, m, c):
   """
    绘制数据点和拟合直线
    
    参数:
        x: x坐标数组
        y: y坐标数组
        m: 斜率
        c: 截距
    
    返回:
        fig: matplotlib图像对象
    """
   fig, ax = plt.subplots()
   ax.scatter(x, y, label='实验数据')
   y_fit = m*x + c
   ax.plot(x, y_fit, 'r', label='拟合直线')
   ax.set_xlabel('频率 (Hz)')
   ax.set_ylabel('电压 (V)')
   ax.legend()
   return fig
    

def calculate_planck_constant(m):
    """
    计算普朗克常量
    
    参数:
        m: 斜率 (单位：V/Hz)
        
    返回:
        h: 计算得到的普朗克常量值 (单位：J·s)
        relative_error: 与实际值的相对误差(%)
    """
    e = 1.602e-19  # 电子电荷 (单位：C)
    h_actual = 6.62607015e-34  # 普朗克常量标准值
    
    h = m * e  # 根据公式 V=(h/e)ν-φ 推导得到 h = m*e
    relative_error = abs(h - h_actual) / h_actual * 100
    
    return h, relative_error

def main():
   """主函数"""
    # 数据文件路径
    filename = "millikan.txt"
    
    # 加载数据
    x, y = load_data(filename)
    
    # 计算拟合参数
    m, c, Ex, Ey, Exx, Exy = calculate_parameters(x, y)
    
    # 打印结果
    print("统计量计算结果:")
    print(f"Ex  = {Ex:.6e}")
    print(f"Ey  = {Ey:.6e}")
    print(f"Exx = {Exx:.6e}")
    print(f"Exy = {Exy:.6e}\n")
    print(f"斜率 m = {m:.6e} V/Hz")
    print(f"截距 c = {c:.6e} V\n")
    
    # 绘制数据和拟合直线
    fig = plot_data_and_fit(x, y, m, c)
    
    # 计算普朗克常量
    h, relative_error = calculate_planck_constant(m)
    print(f"计算得到的普朗克常量 h = {h:.6e} J·s")
    print(f"与实际值的相对误差: {relative_error:.2f}%")
    
    # 保存图像
    fig.savefig("millikan_fit.png", dpi=300)
    plt.show()
if __name__ == "__main__":
    main()
