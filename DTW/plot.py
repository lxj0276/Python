import pandas as pd
from pylab import mpl

from strategy import OPT


mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False      # 解决保存图像是负号'-'显示为方块的问题


if __name__ == '__main__':
    name = "{}{}{}".format(OPT['trend_win'], OPT['return_win'], len(OPT['pos']))
    data = pd.read_csv(OPT['data_dir'], engine='python', index_col=0)
    nav1 = pd.read_csv('output/hs300_nav_{}.csv'.format(name), header=None, index_col=0)
    nav2 = pd.read_csv('output/hs300_nav_my_{}.csv'.format(name), header=None, index_col=0)

    # 基准
    base = pd.read_csv('data/3145.csv', index_col=0)
    base = base.reindex(nav1.index)['close']
    base = base / base[0]

    # data = data.reindex(nav1.index)
    # data = data / data.iloc[0, :]
    # returns = (data / data.shift(1)).fillna(1)
    # base = returns.sum(axis=1) / 29
    # base = base.cumprod()

    df = pd.DataFrame(columns=['hs300', '策略1', '策略2'])
    df.iloc[:, 0] = base
    df.iloc[:, 1] = nav1[1]
    df.iloc[:, 2] = nav2[1]
    df.index = pd.to_datetime(df.index)

    import matplotlib.pyplot as plt

    df.plot()
    plt.savefig('fig/{}.png'.format(name))


