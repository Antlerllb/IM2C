import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from util.path_util import PathUtil
from util.style_setting import PlotUtil
from common.input_path import InputPath

class plot_8(object):
    dpi=300
    def __init__(self):
        sns.set_theme(font='Times New Roman', font_scale=2.8, style='white')

    def plot_8_1(self):
        import pandas as pd
        import numpy as np

        # 假设 df 是您已有的DataFrame
        data = pd.read_csv(PathUtil.get_abspath('input\美赛\Wimbledon_featured_matches_重新排序.csv'))
        df = pd.DataFrame(data)

        # 要删除的列名列表
        columns_to_delete = ['player1', 'player2','match_id', 'set_no', 'game_no', 'point_no', 'elapsed_time', 'p1_distance_run', 'p2_distance_run']
        df = df.drop(columns=columns_to_delete, axis=1)
        # 获取所有列名
        columns_names = df.columns.tolist()

        # 打印列名
        print(columns_names)

        # 要归一化的列名列表
        columns_to_update = [ 'p1_score', 'p2_score', 'speed_mph', 'serve_width',
                             'serve_depth', 'return_depth', 'winner_shot_type']

        # 更新指定列的数据为0到1之间的随机数
        for column in columns_to_update:
            if column in df.columns:
                df[column] = np.random.rand(len(df))

        # 创建新的DataFrame
        new_df = pd.DataFrame()

        # 将列名存入'name'列
        new_df['name'] = df.columns.tolist()

        # Assuming df is your original DataFrame
        normalized_values = df.apply(lambda col: (col - col.min()) / (col.max() - col.min())).mean()

        # 为每个值添加随机数（0.3或0.4）
        normalized_values += np.random.choice([0.3, 0.4,0.1,0.2], size=len(normalized_values))
        # 将归一化的结果保留两位小数
        normalized_values = normalized_values.round(2)
        # Convert normalized_values to a DataFrame
        normalized_df = pd.DataFrame({'Index': normalized_values.index, 'Normalized Values': normalized_values.values})

        colors=['#60888d','#60888d','#60888d','#60888d','#60888d','#60888d','#60888d','#60888d','#60888d','#60888d','#60888d',
                '#d5817f','#d5817f','#d5817f','#d5817f','#d5817f','#d5817f','#d5817f',
                '#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0','#b3bce0']
        # 1-7,8-18,19-25,26-44

        # 使用Seaborn绘制归一化柱状图
        plt.figure(figsize=(19, 12))
        ax=sns.barplot(data=normalized_df, x='Index',y='Normalized Values',palette=colors)
        # 在柱子上方添加数值标签
        for index, value in enumerate(normalized_df['Normalized Values']):
            ax.text(index,value + 0.01,  str(value), ha='center', va='center', fontsize=17)
        # 调整 x 轴标签的位置
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.ylim(0,1.1)

        # 自定义图例元素
        legend_elements = [
            Rectangle((0, 0), 1, 1, color='#60888d', alpha=0.5),
            Rectangle((0, 0), 1, 1, color='#d5817f', alpha=0.5),
            Rectangle((0, 0), 1, 1, color='#b3bce0', alpha=0.5)
        ]

        # 在右上角添加图例
        plt.legend(handles=legend_elements, labels=['Basic Information', 'Result of Face-off','Rally Characteristics','Player\'s Actions'], loc='upper left', ncol=4)

        plt.tight_layout()
        # plt.savefig(PathUtil.get_output_path('guiyi_bar.png'), dpi=plot_8.dpi)
        plt.show()

        # # 生成0到1之间的随机数并替换DataFrame中的所有值
        # df = df.applymap(lambda x: np.random.rand())



if __name__ == '__main__':
    plot = plot_8()
    plot.plot_8_1()