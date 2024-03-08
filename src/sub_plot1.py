import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from util.path_util import PathUtil
from util.style_setting import PlotUtil
from common.input_path import InputPath

class plot_8(object):
    dpi=300
    def __init__(self):
        PlotUtil.set_theme()

    def plot_8_1(self):
        import numpy as np
        import matplotlib.pyplot as plt

        # 设置雷达图的标签
        categories = ['Ace', 'Double_fault', 'Break_pt_won', 'Distance_run', 'Net_pt_won','Break_pt_missed']

        # 设置数据
        data = {
            'Player 1': [1, 0.2, 0.3, 0.7, 0.6,0.1],
            'Player 2': [0.7, 0.4, 0.5, 0.6, 0.55,0.2]
        }

        # 计算角度
        num_categories = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        # 绘制雷达图
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        # 循环绘制每个玩家的雷达图
        for player, values in data.items():
            ax.plot(angles, values + values[:1], label=player, marker='o')
            ax.fill(angles, data['Player 1'] + data['Player 1'][:1], 'b', alpha=0.1)
            ax.fill(angles, data['Player 2'] + data['Player 2'][:1], 'r', alpha=0.1)
        # 添加标签和标题
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        # ax.set_title('Radar Chart Example')

        # 显示图例
        ax.legend(bbox_to_anchor=(0, 0.25, 1.1, 1))
        #

        # 调整布局，确保坐标轴标签不被截掉
        plt.tight_layout()

        # 设置ticks与轴的距离
        plt.tick_params(axis='x', which='both', pad=15)

        # 显示图形
        plt.show()


if __name__ == '__main__':
    plot = plot_8()
    plot.plot_8_1()