import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class plot_8(object):
    dpi=300
    def __init__(self):
        sns.set_theme(font='Times New Roman', font_scale=1.2, style='white')

    @staticmethod
    def right_plot():
        # 创建自定义坐标
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        x, y = np.meshgrid(x, y)

        # 计算双曲抛物面的z值
        z = (y ** 2 / 1 ** 2) - (x ** 2 / 2 ** 2)

        # 创建3D图像
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制曲面
        surf = ax.plot_surface(x, y, z, cmap='gist_rainbow')

        # ax.grid(False)
        # ax.xaxis.pane.fill = False
        # ax.yaxis.pane.fill = False
        # ax.zaxis.pane.fill = False

        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.11)

        # 设置坐标轴标签
        ax.set_xlabel('Shot Distance')
        ax.set_ylabel('Pass Distance')
        # ax.set_zlabel('Momentum')
        # 在图上添加自定义文本
        ax.text(1.13, 1.1, 1.18, "Momentum", color='black', ha='center', va='center')
        plt.show()

if __name__ == '__main__':
    plot = plot_8()
    plot.right_plot()