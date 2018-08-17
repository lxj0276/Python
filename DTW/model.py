import pandas as pd
import numpy as np
from dtw import RConsole


class Model:
    r = RConsole()

    # this is the class to deal with the training result
    class Result:
        def __init__(self, raw, method, past_return):
            self.raw = raw
            self.min_index = raw.idxmin()

            if method == 'filter':
                self.rank = raw.min().sort_values()
            else:
                columns = self.raw.columns
                returns = {columns[i]: past_return[self.min_index[i]][columns[i]]
                           if self.min_index[i] is not np.nan else np.nan
                           for i in range(len(self.min_index))}
                self.rank = pd.Series(returns).sort_values(ascending=False)

        def filter(self, n):
            # filter first n obs.
            n = len(self.rank) if n > len(self.rank) else n
            min_index = self.min_index[self.rank[:n].index]
            return self.rank[:n], min_index

    def __init__(self, data, n=3, m=1):
        if type(data.index) != pd.DatetimeIndex:
            data.index = pd.to_datetime(data.index)

        # 生成月度的索引序列，并筛选掉最后一个月
        month = data.index.year * 100 + data.index.month
        month_filter = month != month[-1]
        month = month[month_filter]
        data = data[month_filter]

        # 生成历史每三个月片段的起止点，按月滚动
        starts = data.index[month.to_series().diff() != 0]  # 每月月初
        assert len(starts) > n+m, '输入数据长度不足 {} 月'.format(n)

        last_day = starts[n: -m]                            # 每三月的最后一个交易日
        first_day = starts[: -n-m]                          # 每三月的第一个交易日
        return_day = starts[n+m:]

        # 生成训练集的所有片段并归一化
        past_trend = [data[first_day[i]: last_day[i]][:-1]
                      for i in range(len(first_day))]
        past_trend = [i / i.iloc[0, ] for i in past_trend]
        past_return = [data.loc[return_day[i]] / data.loc[last_day[i]]
                       for i in range(len(last_day))]

        # 生成测试集，最近三个月的数据
        unique = month.unique()
        now = data[month >= unique[-n]]
        now = now / now.iloc[0, ]                           # 归一化

        self.now = now
        self.past_trend = {last_day[i].strftime('%Y-%m-%d'): past_trend[i]
                           for i in range(len(last_day))}
        self.past_return = {last_day[i].strftime('%Y-%m-%d'): past_return[i]
                            for i in range(len(last_day))}

    def train(self, method='filter'):
        past_return, past_trend, now = self.past_return, self.past_trend, self.now
        r = self.r

        result = pd.DataFrame(index=past_return.keys(), columns=now.columns)
        # for each stock
        for i in range(now.shape[1]):
            if method == 'non_filter':
                col = {key: r.dtw(past_trend[key].iloc[:, i], now.iloc[:, i]) for key in past_trend}
                result.iloc[:, i] = pd.Series(col)
            elif method == 'filter':
                filtered = {key: past_return[key][i] for key in past_return
                            if past_return[key][i] > 1}
                col = {key: r.dtw(past_trend[key].iloc[:, i], now.iloc[:, i]) for key in filtered}
                result.iloc[:, i] = pd.Series(col)
            else:
                raise Exception("没有定义的方法：{}".format(method))

        return self.Result(result, method, self.past_return)


if __name__ == '__main__':
    industry = pd.read_csv('data/中信一级行业指数/industry.csv', engine='python', index_col=0)

    # 关键代码
    model = Model(industry, 6, 3)
    results = model.train()
    ranks, ends = results.filter(28)
    print(ranks, '\n', ends)

    # 画图展示结果
    stock = ends.index[-1]
    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1)
    model.now[stock].plot()
    plt.subplot(1, 2, 2)
    model.past_trend[results.min_index[stock]][stock].plot()
    plt.show()
