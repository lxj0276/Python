import pandas as pd

from Back import Context
from model import Model
from time import time


def signal(context, date, refresh):
    if date in refresh:
        print("{}调仓".format(date))
        close = context.GlobalParam['daily_close']
        begin = time()

        model = Model(close[:date], 3, 1)
        result = model.train()
        ranks, _ = result.filter(5)

        print("耗时{:.2f}秒".format(time()-begin))
        weight = pd.Series(0.2, index=ranks.index)
        weight = weight.reindex(close.columns).fillna(0)
        return True, weight
    else:
        return False, None


def main():
    data = pd.read_csv('data/中信一级行业指数/industry.csv', engine='python', index_col=0)
    date_index = pd.to_datetime(data.index)

    # 生成月度的索引序列
    month = date_index.year * 100 + date_index.month
    starts = date_index[month.to_series().diff() != 0]  # 每月月初
    cut = month.unique() >= 201001
    cut[month.unique() >= 201808] = False
    starts = [i.strftime('%Y-%m-%d') for i in starts[cut]]

    context = Context()
    context.Add['signal_params'] = {'refresh': starts}
    context.BktestParam['signal'] = signal
    context.GlobalParam['daily_close'] = data
    context.GlobalParam['daily_dates'] = data.loc[starts[0]: starts[-1]].index.tolist()

    begin = time()

    w = context.cal_long_weight()
    print(w)
    nav = context.cal_nav()
    print(nav)

    print("total time : {:.2f}".format(time()-begin))

    import matplotlib.pyplot as plt
    data.index = pd.to_datetime(data.index)
    nav.index = pd.to_datetime(nav.index)
    data.plot()
    nav.plot()
    plt.show()


if __name__ == '__main__':
    main()
