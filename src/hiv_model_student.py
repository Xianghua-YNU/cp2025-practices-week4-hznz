import numpy as np
import matplotlib.pyplot as plt

class HIVModel:
    def __init__(self, A, alpha, B, beta):
        """
        初始化HIV模型参数。
        :param A: 模型参数A
        :param alpha: 模型参数α
        :param B: 模型参数B
        :param beta: 模型参数β
        """
        self.A = A
        self.alpha = alpha
        self.B = B
        self.beta = beta

    def viral_load(self, time):
        """
        计算病毒载量。
        :param time: 时间序列
        :return: 病毒载量数组
        """
        return self.A * np.exp(-self.alpha * time) + self.B * np.exp(-self.beta * time)

    def plot_model(self, time, label=None):
        """
        绘制模型曲线。
        :param time: 时间序列
        :param label: 曲线标签
        """
        viral_load = self.viral_load(time)
        plt.plot(time, viral_load, label=label)
        plt.xlabel('Time (days)')
        plt.ylabel('Viral Load')
        plt.title('HIV Viral Load Over Time')
        plt.legend()
        plt.grid(True)


def load_hiv_data(filepath):
    """
    加载HIV实验数据。
    :param filepath: 数据文件路径
    :return: 时间数组和病毒载量数组
    """
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        return data['time_in_days'], data['viral_load']
    else:
        return np.loadtxt(filepath, delimiter=',', unpack=True)


def main():
    # 1.1 探索模型
    # 生成时间序列
    time_model = np.linspace(0, 10, 100)

    # 初始化不同参数的模型
    model1 = HIVModel(A=1000, alpha=0.5, B=500, beta=0.1)
    model2 = HIVModel(A=800, alpha=0.4, B=600, beta=0.2)
    model3 = HIVModel(A=1200, alpha=0.6, B=400, beta=0.05)

    # 绘制模型曲线
    plt.figure(figsize=(10, 6))
    model1.plot_model(time_model, label='Model 1: A=1000, alpha=0.5, B=500, beta=0.1')
    model2.plot_model(time_model, label='Model 2: A=800, alpha=0.4, B=600, beta=0.2')
    model3.plot_model(time_model, label='Model 3: A=1200, alpha=0.6, B=400, beta=0.05')
    plt.show()

    # 1.2 拟合实验数据
    # 加载实验数据
    try:
        time_data, viral_load_data = load_hiv_data('data/HIVseries.npz')
    except FileNotFoundError:
        time_data, viral_load_data = load_hiv_data('HIVseries.csv')

    # 初始化模型参数（手动调整）
    model_fit = HIVModel(A=1000, alpha=0.5, B=500, beta=0.1)

    # 绘制模型曲线和实验数据点
    plt.figure(figsize=(10, 6))
    model_fit.plot_model(time_model, label='Fitted Model: A=1000, alpha=0.5, B=500, beta=0.1')
    plt.scatter(time_data, viral_load_data, color='red', label='Experimental Data', zorder=5)
    plt.xlabel('Time (days)')
    plt.ylabel('Viral Load')
    plt.title('HIV Viral Load Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 1.3 分析结果
    # 假设 alpha 是 T 细胞感染率的倒数
    alpha = model_fit.alpha
    latency_period = 10  # 年
    latency_period_days = latency_period * 365  # 转换为天数

    # 计算 T 细胞感染率的倒数
    t_cell_infection_rate = 1 / alpha

    print(f"T细胞感染率的倒数 (1/α): {t_cell_infection_rate:.2f} 天")
    print(f"HIV潜伏期: {latency_period_days} 天")

    if t_cell_infection_rate < latency_period_days:
        print("T细胞感染率的倒数小于HIV潜伏期，表明感染率较高。")
    else:
        print("T细胞感染率的倒数大于HIV潜伏期，表明感染率较低。")


if __name__ == "__main__":
    main()
