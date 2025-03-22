import numpy as np
import matplotlib.pyplot as plt

class HIVModel:
    def __init__(self, A, alpha, B, beta):
        # TODO: 初始化模型参数
        self.A = A
        self.alpha = alpha
        self.B = B
        self.beta = beta

    def viral_load(self, time):
        # TODO: 计算病毒载量
        return self.A * np.exp(-self.alpha * time) + self.B * np.exp(-self.beta * time)
     
    def plot_model(self, time,data_time=None, data_load=None):
        # TODO: 绘制模型曲线
        viral_load=self.viral_load(time)
        plt.plot(time, viral_load, label=f'Model (α={self.alpha}, β={self.beta})')
        if data_time is not None and data_load is not None:
            plt.scatter(data_time, data_load, color='red', s=20, label='Experimental Data')  # s控制点大小
        
        plt.xlabel('Time(day)')
        plt.ylabel('Viral Load')
        plt.title("HIV Viral Load Model")
        plt.legend()
        plt.grid(True)
        plt.show()

def load_hiv_data(filepath):
    # TODO: 加载HIV数据
    try:
        data = np.loadtxt(filepath)
        return data['time_in_days'], data['viral_load']
    except:
        data = np.loadtxt(filepath, delimiter=',')
        return data[:, 0], data[:, 1]
    
def main():
    # TODO: 主函数，用于测试模型
    model = HIVModel(A=100000, alpha=0.35, B=61000, beta=1.0)

    # 生成时间序列
    time = np.linspace(0, 10, 100) #10天，均分100时间间隔


    # 加载实验数据
    try:
        data_time, data_load = load_hiv_data('data/HIVseries.csv')
    except FileNotFoundError:
        print("未找到数据文件，仅绘制模型曲线")
        data_time, data_load = None, None

    model.plot_model(time, data_time=data_time, data_load=data_load)

    latent_period = 1 / model.alpha  # 以天为单位
    print(f"T细胞感染率倒数 1/α = {latent_period:.16f} 天")
    print(f"T细胞感染率的倒数1/α与十年潜伏期比值为 {latent_period/3650:.16f}")

    model.plot_model(time) #绘制模型图

    plt.scatter(data_time, data_load)
    plt.title("HIV Experimental Data")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
