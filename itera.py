import numpy as np
import matplotlib.pyplot as plt


# 解决思路：
# 1. 初始化一个场分布，通常使用高斯分布作为初始猜测。
# 2. 定义矩形谐振腔的参数，如腔长、腔宽、波长等。
# 3. 根据菲涅尔-基尔霍夫衍射积分公式，通过迭代更新场分布。
# 4. 检查收敛性，当满足收敛条件时停止迭代。


# 矩形谐振腔的参数
Lx = 1  # 腔长 (x 方向)
Ly = 0.5  # 腔宽 (y 方向)
wavelength = 0.6328e-6  # 波长
k = 2 * np.pi / wavelength  # 波数
Nx = 512  # x 方向的采样点数
Ny = 256  # y 方向的采样点数
x = np.linspace(-Lx / 2, Lx / 2, Nx)  # x 方向的坐标
y = np.linspace(-Ly / 2, Ly / 2, Ny)  # y 方向的坐标
dx = x[1] - x[0]  # x 方向的采样间隔
dy = y[1] - y[0]  # y 方向的采样间隔
X, Y = np.meshgrid(x, y)  # 生成二维网格


# 初始场分布，这里使用高斯分布作为初始猜测
w0x = 0.1  # x 方向的初始束腰半径
w0y = 0.05  # y 方向的初始束腰半径
u0 = np.exp(-(X ** 2 / w0x ** 2 + Y ** 2 / w0y ** 2))  # 初始场分布


# 迭代参数
num_iter = 1000  # 最大迭代次数
tol = 1e-6  # 收敛阈值
u = u0  # 初始场分布


def fresnel_kernel(k, coord, z, lambda_):
    """
    计算菲涅尔-基尔霍夫衍射积分核函数
    :param k: 波数
    :param coord: 坐标 (x 或 y)
    :param z: 传播距离
    :param lambda_: 波长
    :return: 衍射积分核函数
    """
    return np.exp(1j * k * coord ** 2 / (2 * z)) / (1j * lambda_ * z)


for iter in range(num_iter):
    # 从一个反射镜到另一个反射镜的传播距离，假设为 Lx
    z = Lx
    # 菲涅尔-基尔霍夫衍射积分核函数
    hx = fresnel_kernel(k, X, z, wavelength)
    hy = fresnel_kernel(k, Y, z, wavelength)
    # 对 x 和 y 方向分别进行衍射积分
    u_new = np.fft.fft2(np.fft.fftshift(hx * np.fft.ifft2(np.fft.fftshift(u))))
    u_new = np.fft.ifft2(np.fft.fftshift(hy * np.fft.fft2(np.fft.fftshift(u_new))))
    u_new = u_new * np.exp(1j * k * z)  # 考虑传播相位
    # 归一化
    u_new = u_new / np.max(np.abs(u_new))
    # 检查收敛性
    if np.max(np.abs(u_new - u)) < tol:
        break
    u = u_new


# 显示结果
plt.figure()
plt.imshow(np.abs(u), cmap='hot')
plt.colorbar()
plt.title("The intensity distribution")
plt.show()

