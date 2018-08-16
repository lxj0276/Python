import pandas as pd
from pylab import mpl

from strategy import OPT


mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False      # 解决保存图像是负号'-'显示为方块的问题


if __name__ == '__main__':
    for i in [315, 335, 635, 3110, 3310, 6310]:
        data = pd.read_csv(OPT['data_dir'], engine='python', index_col=0)
        nav1 = pd.read_csv('output/nav_{}.csv'.format(i), header=None, index_col=0)
        nav2 = pd.read_csv('output/nav_my_{}.csv'.format(i), header=None, index_col=0)
        data = data.reindex(nav1.index)
        data = data / data.iloc[0, :]
        returns = (data / data.shift(1)).fillna(1)

        base = returns.sum(axis=1) / 29
        base = base.cumprod()

        df = pd.DataFrame(columns=['基准', '策略1', '策略2'])
        df.iloc[:, 0] = base
        df.iloc[:, 1] = nav1[1]
        df.iloc[:, 2] = nav2[1]
        df.index = pd.to_datetime(df.index)
        # data = data[['CI005027.WI', 'CI005016.WI', 'CI005015.WI', 'CI005026.WI']]
        # data['策略'] = nav[1]
        # print(nav[1])

        import matplotlib.pyplot as plt
        df.plot()
        # plt.show()
        plt.savefig('fig/{}.png'.format(i))