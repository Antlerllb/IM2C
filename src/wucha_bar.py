import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from util.path_util import PathUtil
from util.style_setting import PlotUtil
from common.input_path import InputPath
from matplotlib.patches import Rectangle
from matplotlib.patches import Rectangle, FancyBboxPatch

class plot_8(object):
    dpi=300
    def __init__(self):
        sns.set_theme(font='Times New Roman', font_scale=1.7, style='whitegrid')

    def right_plot(self):
        id = [1, 2,3, 4, 5, 6,7, 8,9,
                      10, 11,12,13, 14,15, 16, 17,18,
                      19, 20,21, 22, 23, 24,25, 26,27,
                      28, 29,30, 31, 32, 33,34, 35,36]
        values = [75, -35, 0,45,20,0,50,-17,0,
                  30,7,0,62,-33,0,75,-36,0,
                  55,-29,0,76,21,0,43,-17,0,
                  45,-19,0,78,-15,0,66,-22,0]
        errors = [3,5,0, 4,5,0,4,3,0,
                  3, 5,0, 4,5,0,4,3,0,
                  3, 5,0, 4,5,0,4,3,0,
                  3, 5,0, 4,5,0,4,3,0]

        # 创建数据框
        data = {'id': id, 'Values': values, 'Errors': errors}
        df = pd.DataFrame(data)

        # 使用Seaborn绘制带有误差条的柱状图
        plt.figure(figsize=(8, 6))

        momentum_color='#ff6a00'
        none_color='#2b9052'#4b9067
        no_color='red'
        #自定义颜色
        colors=[momentum_color,none_color,no_color,momentum_color,none_color,no_color,momentum_color,none_color,no_color,
                momentum_color,none_color,no_color,momentum_color,none_color,no_color,momentum_color,none_color,no_color,
                momentum_color,none_color,no_color,momentum_color,none_color,no_color,momentum_color,none_color,no_color,
                momentum_color,none_color,no_color,momentum_color,none_color,no_color,momentum_color,none_color,no_color]

        # 使用barplot函数绘制柱状图
        ax=sns.barplot( x='id', y='Values',data=df, ci=None, capsize=0.2, palette=colors,width=1)

        # # 添加误差条
        # for i, value in enumerate(values):
        #     plt.errorbar(x=i, y=value, yerr=errors[i], color='black', capsize=3, capthick=1.5)
        # 添加误差条，仅对id为奇数的柱形显示误差条
        for i, (value, current_id) in enumerate(zip(values, id)):
            if current_id % 3 != 0:
                plt.errorbar(x=i, y=value, yerr=errors[i], color='black', capsize=3, capthick=1.5)

        # 添加标题和标签
        #plt.title('Bar Plot with Error Bars')
        plt.xlabel('')
        plt.ylabel('Momentum(% Change)')
        plt.ylim(-50,100)

        # 禁用 x 轴 ticks
        plt.xticks([])

        # 设置x轴刻度位置和标签
        x_ticks_positions = [0.5,3.5, 6.5,
                             9.5,12.5 ,15.5,
                             18.5,21.5,24.5,
                             27.5,30.5,33.5]  # 你希望放置标签的位置
        x_ticks_labels = ['Slam dunk', 'Fast break', 'Three-pointer',
                          'Hat-trick', 'Clean sheet', 'Penalty kick',
                          'Smash', 'Spin', 'Rally',
                          'Ace', 'Net point', 'Break point']  # 对应位置的标签文本
        plt.xticks(x_ticks_positions, x_ticks_labels)

        # 调整 x 轴标签的位置
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # 自定义图例元素
        legend_elements = [
            Rectangle((0, 0), 1, 1, color='#f57a1a'),
            Rectangle((0, 0), 1, 1, color='#34724b')
        ]

        # 在右上角添加图例
        plt.legend(handles=legend_elements,
                   labels=['Momentum', 'None'],
                    loc='upper left',ncol=2,bbox_to_anchor=(-0.015,1.23))

        plt.tight_layout()
        # 显示图形
        # plt.savefig(PathUtil.get_output_path('wucha_bar.png'), dpi=plot_8.dpi)
        plt.show()


if __name__ == '__main__':
    plot = plot_8()
    plot.right_plot()