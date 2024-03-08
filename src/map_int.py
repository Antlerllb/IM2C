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
        sns.set_theme(font='Times New Roman', font_scale=2, style='whitegrid')

    def right_plot(self):
        # 创建示例数据框
        data = pd.read_csv(PathUtil.get_abspath('input\美赛\Wimbledon_featured_matches_重新排序.csv'))

        df1 = pd.DataFrame(data)
        # 提取4到6列
        df = df1.iloc[:, 9:15]

        # 更改列名
        new_column_names = {'p1_points_won': 'p1_points','p2_points_won': 'p2_points'}
        df.rename(columns=new_column_names, inplace=True)

        clustermap = sns.clustermap(df.corr(), annot=True, cmap="Blues", dendrogram_ratio=0.2,tree_kws={'linewidths': 1.7}, annot_kws={'fontsize': 20}, linewidths=2.3, linecolor='white', fmt='.2f')

        # 获取图的Axes对象
        ax = clustermap.ax_heatmap

        # 将小于0.9的数值设置为空字符串
        for text in ax.texts:
            if float(text.get_text()) <= 0.22:
                text.set_text("")

        # 显示图形
        # plt.savefig(PathUtil.get_output_path('map_int.png'), dpi=plot_8.dpi)
        plt.show()


if __name__ == '__main__':
    plot = plot_8()
    plot.right_plot()