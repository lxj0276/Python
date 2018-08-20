import pandas as pd
from pylab import mpl

from strategy import OPT


mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False      # 解决保存图像是负号'-'显示为方块的问题


if __name__ == '__main__':
    data = pd.read_csv(OPT['data_dir'], engine='python', index_col=0)
    nav1 = pd.read_csv('output/zz800_s1.csv', header=None, index_col=0)
    nav3 = pd.read_csv('output/hs300_nav_align_6163.csv', header=None, index_col=0)

    # 基准
    base = pd.read_csv(OPT['base_dir'], index_col=0)
    base = base.reindex(nav1.index)['close']
    base = base / base[0]

    # data = data.reindex(nav1.index)
    # data = data / data.iloc[0, :]
    # returns = (data / data.shift(1)).fillna(1)
    # base = returns.sum(axis=1) / 29
    # base = base.cumprod()

    df = pd.DataFrame(columns=['hs300', '策略-中证', '策略-沪深'])
    df.iloc[:, 0] = base
    df.iloc[:, 1] = nav1[1]
    # df.iloc[:, 2] = nav2[1]
    df.iloc[:, 2] = nav3[1]
    df.index = pd.to_datetime(df.index)

    import matplotlib.pyplot as plt

    df.plot()
    plt.savefig('fig/zz800.png')


