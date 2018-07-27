import pandas as pd
import numpy as np
import tushare as ts
# import matplotlib.pyplot as plt


class BktestParam:
    init_period = ['2005-03-01', '2018-01-31']
    commission_rate = 0                         # 手续费
    signal = None                               # 交易信号函数
    refresh_dates = None                        # 调仓日期序列
    signal_params = {}                          # 交易信号函数需要的额外参数


class GlobalParam:
    data = None                                 # 数据
    asset_pool = None                           # 资产池
    daily_close = None                          # 资产每日的收盘价
    daily_dates = None                          # 交易日期序列
    base_nav = None                             # 基准的净值曲线


class BktestResult:
    w = None                                    # 每个调仓日期的权重
    nav = None                                  # 回测的净值曲线
    nav_perf = None                             # 回测表现
    df = None                                   # 回测结果


def signal(func):
    BktestParam.signal = func
    return func


def cal_long_weight():
    refresh_dates = []
    BktestResult.w = pd.DataFrame(columns=GlobalParam.asset_pool)
    for date in GlobalParam.daily_dates:
        sig, weight = BktestParam.signal(date, **BktestParam.signal_params)
        if sig:
            refresh_dates.append(date)
            BktestResult.w.loc[date, :] = weight
            BktestResult.w.loc[date, :].fillna(0, inplace=True)

    BktestParam.refresh_dates = refresh_dates


def cal_nav():
    close = GlobalParam.daily_close
    w = BktestResult.w
    commission_rate = BktestParam.commission_rate
    daily_dates = GlobalParam.daily_dates
    refresh_dates = BktestParam.refresh_dates

    # 建仓
    nav = pd.Series(index=daily_dates)
    refresh_w = w.iloc[0, :]
    start_date = w.index.tolist()[0]
    nav[start_date] = 1 - commission_rate
    last_portfolio = np.asarray((1 - commission_rate) * refresh_w)
    for i in range(daily_dates.index(start_date)+1, len(daily_dates)):
        # 执行净值更新，分空仓和不空仓情形
        if sum(last_portfolio) == 0:
            nav[i] = nav[i-1]
        else:
            last_portfolio = np.asarray(close.iloc[i, :] / close.iloc[i - 1, :]) * last_portfolio
            nav[i] = np.nansum(last_portfolio)

        # 判断当前日期是否为新的调仓日，是则进行调仓
        if daily_dates[i] in refresh_dates:
            # 将最新仓位归一化
            last_w = np.zeros_like(last_portfolio) if np.nansum(last_portfolio) == 0 \
                else last_portfolio

            # 获取最新的权重分布
            refresh_w = w.loc[daily_dates[i], :]

            # 根据前后权重差别计算换手率，调整净值
            refresh_turn = np.nansum(np.abs(refresh_w - last_w))
            nav[i] = nav[i] * (1 - refresh_turn * commission_rate)
            last_portfolio = np.asarray(nav[i] * refresh_w)

    BktestResult.nav = nav


def main():
    hs300 = ts.get_hist_data('hs300')
    sz50 = ts.get_hist_data('sz50')
    data = pd.concat([hs300['close'], sz50['close']], axis=1)
    data.columns = ['hs300', 'sz50']
    data = data[::-1] / data[::-1].iloc[0, :]

    # init global params
    GlobalParam.data = data
    GlobalParam.daily_dates = data.index.tolist()
    GlobalParam.daily_close = data
    GlobalParam.asset_pool = ['hs300', 'sz50']

    # init back test params
    @signal
    def func(date):
        raw = GlobalParam.data
        weight = raw.loc[date, :].sort_values()
        weight = pd.Series([1, 0], index=weight.index)
        sig = GlobalParam.daily_dates.index(date) % 10 == 0

        return sig, weight

    cal_long_weight()
    cal_nav()

    data['test'] = BktestResult.nav
    # data.plot()
    # plt.show()
    # print(BktestResult.nav)
    print(BktestResult.w)
    # print(data)


if __name__ == '__main__':
    main()