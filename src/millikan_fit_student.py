"""
最小二乘拟合和光电效应实验
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """
    加载数据文件（带异常处理）
    
    参数:
        filename: 数据文件路径
        
    返回:
        x: 频率数据数组
        y: 电压数据数组
        
    异常:
        FileNotFoundError: 文件不存在
        ValueError: 数据格式错误
    """
    try:
        data = np.loadtxt(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"错误：数据文件 '{filename}' 未找到") from None
    except Exception as e:
        raise ValueError(f"数据文件读取失败: {str(e)}") from None
    
    if data.shape[1] != 2:
        raise ValueError("数据文件必须包含且仅包含两列数据")
    
    x = data[:, 0]
    y = data[:, 1]
    
    # 验证数据有效性
    if len(x) == 0 or len(y) == 0:
        raise ValueError("数据文件不能为空")
    if len(x) != len(y):
        raise ValueError("x和y数据长度不一致")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("数据包含无效的NaN值")
    
    return x, y

def calculate_parameters(x, y):
    """
    计算最小二乘拟合参数（带异常处理）
    """
    if len(x) != len(y):
        raise ValueError("x和y数据长度不一致")
    
    if len(x) < 2:
        raise ValueError("至少需要2个数据点进行拟合")
    
    try:
        Ex = np.mean(x)
        Ey = np.mean(y)
        Exx = np.mean(x**2)
        Exy = np.mean(x*y)
    except TypeError:
        raise ValueError("输入数据必须为数值类型")
    
    denominator = Exx - Ex**2
    if abs(denominator) < 1e-15:
        raise ValueError("数据x值的方差为零，无法计算斜率")
    
    m = (Exy - Ex * Ey) / denominator
    c = (Exx * Ey - Ex * Exy) / denominator
    
    return m, c, Ex, Ey, Exx, Exy

def plot_data_and_fit(x, y, m, c):
    """
    绘制数据点和拟合直线（带参数验证）
    
    参数:
        x: x坐标数组
        y: y坐标数组
        m: 斜率
        c: 截距
    
    返回:
        fig: matplotlib图像对象
    """
    # 参数类型验证
    if not all(isinstance(v, (int, float)) for v in [m, c]):
        raise TypeError("斜率和截距必须是数值类型")
    
    fig = plt.figure(figsize=(8, 6))
    try:
        plt.scatter(x, y, color="blue", label="实验数据", zorder=10)
        x_fit = np.linspace(np.min(x), np.max(x), 100)
        y_fit = m * x_fit + c
        plt.plot(x_fit, y_fit, color="red", linewidth=1.5, label="拟合直线")
        plt.xlabel("频率 ν (Hz)", fontsize=12)
        plt.ylabel("电压 V (V)", fontsize=12)
        plt.title("光电效应实验数据与最小二乘拟合", fontsize=14)
        plt.legend()
        plt.grid(linestyle=":", alpha=0.6)
    except Exception as e:
        plt.close()
        raise RuntimeError(f"绘图失败: {str(e)}") from None
    
    return fig

def calculate_planck_constant(m):
    """
    计算普朗克常量（带参数验证）
    
    参数:
        m: 斜率
        
    返回:
        h: 计算得到的普朗克常量值
        relative_error: 与实际值的相对误差(%)
    """
    if not isinstance(m, (int, float)):
        raise TypeError("斜率必须是数值类型")
    if m <= 0:
        raise ValueError("斜率必须为正数")
    
    e = 1.602e-19
    h_actual = 6.62607015e-34
    
    try:
        h = m * e
        relative_error = abs(h - h_actual) / h_actual * 100
    except Exception as e:
        raise RuntimeError(f"计算普朗克常量失败: {str(e)}") from None
    
    return h, relative_error

def main():
    """主函数（带全局异常处理）"""
    try:
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
        fig.savefig("millikan_fit.png", dpi=300, bbox_inches="tight")
        plt.show()
    
    except Exception as e:
        print(f"\n程序运行失败: {str(e)}")
        print("请检查：")
        print("1. 数据文件是否存在且格式正确")
        print("2. 数据包含有效的数值")
        print("3. 数据满足最小二乘拟合要求")
        exit(1)

if __name__ == "__main__":
    main()
