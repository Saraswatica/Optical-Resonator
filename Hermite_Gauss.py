import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite


def main():
    # 初始化
    num = 1
    # 设置光束参数
    wx = 1  # x 方向束腰宽度，可根据实际情况调整
    wy = 1  # y 方向束腰宽度，可根据实际情况调整
    fig, axs = plt.subplots(3, 3)
    for m in range(1, 4):
        for n in range(1, 4):
            # 定义 x 和 y 轴的范围和采样点数
            x = np.linspace(-5 * wx, 5 * wx, 100)
            y = np.linspace(-5 * wy, 5 * wy, 100)
            X, Y = np.meshgrid(x, y)
            # 计算厄米多项式
            hermitePolyX = hermite(m - 1)(np.sqrt(2) * X / wx)
            hermitePolyY = hermite(n - 1)(np.sqrt(2) * Y / wy)
            # 计算光场复振幅（这里忽略了一些不影响相对强度分布的常数因子）
            U = hermitePolyX * hermitePolyY * np.exp(-(X ** 2) / (wx ** 2) - (Y ** 2) / (wy ** 2))
            # 计算强度分布
            I = np.abs(U) ** 2
            # 绘制强度分布图
            ax = axs[(num - 1) // 3, (num - 1) % 3]
            ax.imshow(I)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'2D Hermite-Gaussian Beam (m = {m-1}, n = {n-1})')
            num += 1


if __name__ == "__main__":
    main()
    plt.show()