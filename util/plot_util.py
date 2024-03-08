# encoding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def init_style():
    """
    根据主机环境设置 Seaborn 主题。
    """
    plt.clf()
    sns.reset_orig()
    sns.set_theme(font='Times New Roman', style='whitegrid')
    plt.rcParams['figure.dpi'] = 300
