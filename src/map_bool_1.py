import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.ma import MaskedArray

from util.path_util import PathUtil
from util.style_setting import PlotUtil
from common.input_path import InputPath

class plot_8(object):
    dpi = 300
    def __init__(self):
        sns.set_theme(font='Times New Roman', font_scale=1.8, style='white')

    @staticmethod
    def right_plot():
        # 读取CSV文件
        data = pd.read_csv(PathUtil.get_abspath('input\美赛\Wimbledon_featured_matches_重新排序.csv'))

        # 删除第四列
        df = data.drop(columns=['winner_shot_type'])

        # 提取28到45列
        df = df.iloc[:, 27:45]
        correlation_matrix = df.corr()

        # 获取数据框中所有的列名
        columns = correlation_matrix.columns

        # 初始化相关性矩阵
        correlation_matrices = []

        a=[4,5,2,3,8,9,12,13]
        b=[0,1,6,7,10,11,14,15,16,17]

        # 计算每两列之间的相关性矩阵，奇数列只与奇数列聚类，偶数列只与偶数列聚类
        for i in a:
            for j in b:
                col1, col2 = columns[i], columns[j]
                selected_columns = [col1, col2]
                correlation_matrix = df[selected_columns].corr()
                correlation_matrices.append((col1, col2, correlation_matrix.iloc[0, 1]))


        # 将相关性矩阵的结果组合成一个数据框
        result_df = pd.DataFrame(correlation_matrices, columns=['Column1', 'Column2', 'Correlation'])

        # 使用Seaborn绘制热图
        plt.figure(figsize=(10, 8))

        # 设置颜色条的属性
        cbar_kws = {  # 颜色条方向，可以是'horizontal'或'vertical'
            'ticks': [-0.25,0,0.25,0.5,0.75,1],  # 颜色条上的刻度值 # 颜色条的长度缩放因子
        }

        pivot_result = result_df.pivot(index='Column1', columns='Column2', values='Correlation')
        # 按照指定的索引顺序排序
        index_order = ['p1_winner','p2_winner','p1_net_pt','p2_net_pt','p1_double_fault','p2_double_fault','p1_break_pt','p2_break_pt']
        pivot_result = pivot_result.reindex(index_order)
        # 按照指定的列索引顺序排序
        column_order = ['p1_ace','p2_ace', 'p1_unf_err', 'p2_unf_err','p1_net_pt_won','p2_net_pt_won','p1_break_pt_won','p2_break_pt_won','p1_break_pt_missed','p2_break_pt_missed']
        pivot_result= pivot_result[column_order]

        clustermap=sns.clustermap(pivot_result, cmap='Oranges', annot=True, annot_kws={'fontsize': 14.5}, square=True, linewidths=2, linecolor='white',
                                    tree_kws={'linewidths': 1.4}, cbar_kws=cbar_kws,cbar_pos=(0.06, 0.82, 0.02, 0.17))
        # #
        # 获取图的Axes对象
        ax = clustermap.ax_heatmap

        # 将小于0.9的数值设置为空字符串
        for text in ax.texts:
            if float(text.get_text()) < -0.02:
                text.set_text("")

        plt.xlabel('')  # 不显示 x 轴标签
        plt.ylabel('')  # 不显示 y 轴标签
        # plt.savefig(PathUtil.get_output_path('map_bool.png'), dpi=plot_8.dpi)
        plt.show()



if __name__ == '__main__':
    plot = plot_8()
    plot.right_plot()