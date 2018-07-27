import pandas as pd
import numpy as np
import FactorModel as Fm
import backtest as bt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pylab import mpl


# init back test params
@bt.signal
def func(date, risk):
    monthly_dates = bt.GlobalParam.monthly_dates
    if date in monthly_dates:
        now = monthly_dates.index(date)
        if now < 12:
            return False, None

        now_data = bt.GlobalParam.monthly_return_data.iloc[now - 12:now, :].dropna(axis=1)
        predicts = now_data.ewm(alpha=0.8).mean().iloc[-1]
        rs = np.cov(np.asmatrix(now_data).T)

        r = (1 + risk) ** (1 / 12) - 1
        up = predicts.iloc[-1] / r + 0.1
        up = up if up < 1.0 else 1.0
        try:
            # 风险平价
            w = Fm.min_risk(returns=predicts, risk_structure=rs, target_return=r, up=up)
            w = pd.Series(w, index=now_data.columns)
            w = w.reindex(bt.GlobalParam.asset_pool)
            # 动量因子
            w = factor_adj(w, date, **bt.BktestParam.momentum_params)
            # RSRS
            w = rs_adj(w, date, **bt.BktestParam.rs_params)
            return round(np.nansum(w), 1) == 1.0, np.asarray(w)
        except ValueError:
            return False, None
    else:
        old = bt.BktestResult.w.iloc[-1, :]
        new = rs_adj(old, date, **bt.BktestParam.rs_params)
        return np.nansum(np.abs(old - new)) == 0, new


def factor_adj(w, date, k):
    """
    依赖于经过标准化后的
    数据1：每列对应每个资产使用的动量因子
    数据2：每列对应每个资产使用的估值因子 【见对照表】
    对每一类资产：
        取出当期的动量因子值，与过去一年【12个月】平均值比较，大于则权重更新乘以 K， 否则更新为 1/K
        取出当期的估值因子值，与2005年至当期所有已知因子的均值比较，与上文采取同样的调整方式
    计算完毕，将权重归一化
    """
    monthly_index = bt.GlobalParam.monthly_index
    momentum_data = bt.GlobalParam.momentum_data
    now = monthly_index.index(date)
    sig = momentum_data.iloc[now, :] - momentum_data.iloc[now-12:now, :].mean()
    sig[sig <= 0] = 1 / k
    sig[sig > 0] = k
    sig = sig.reindex(bt.GlobalParam.asset_pool)
    w = (sig * w * w[:-1].sum() / sig.sum()).fillna(w[-1])

    return w


def rs_adj(w, date, m, s):
    """
    数据3：回测区间以日为单位的最高价最低价序列用前N日回归后的每日斜率序列以及R方
    对每一列资产计算RSRS指标：
        取前M日的斜率时间序列，计算当日的标准分数Z
        用Z乘以R方，作为当日RSRS的指标
        根据资产对应的买卖信号的阈值进行判断：
            若为买信号则维持原来持有
            若为卖信号则将仓位更新为0，增加现金的权重
    1.资产close增加一列：现金，净值无变化，全为1 : daily_close
    2.资产池增加现金 : cash
    3.权重增加一列现金的权重，在风险平权时不计算，而是在后贴0
    4.daily_days 风险平权、factor_adj()月度调仓调用得到每日权重，但rs_adj()每天调用
        逢月底大调：建立一个月底日期表
        每日小调：若为买入信号则维持当月不变，卖出信号则进行调整现金
    """
    daily_dates = bt.GlobalParam.daily_dates
    rs_data = bt.GlobalParam.rs_data
    now = daily_dates.index(date)

    sig = pd.Series(index=bt.GlobalParam.asset_pool[:-1])
    for i in sig.index:
        arr = rs_data[i].iloc[max(now - m, 0):now+1, 0]
        z = (arr - arr.mean()) / arr.std()
        sig[i] = rs_data[i]['r2'] * z[-1]

    sig = sig - s
    sig = sig.reindex(bt.GlobalParam.asset_pool).fillna(1)
    w[sig < 0] = 0
    w[-1] = 1 - w[:-1].sum()

    return w


def init_data():
    assets = [3145, 3159, 4978, 6455, 14599]
    data_list = [pd.read_csv('data/{}.csv'.format(i), index_col='Date')
                 for i in assets]
    data = pd.DataFrame(index=data_list[0].index)
    for i in range(len(assets)):
        data[assets[i]] = data_list[i]['ClosePrice']

    data_close = data.apply(lambda x: x / x.dropna()[0])                            # 各类资产净值曲线
    data_close[0] = 1                                                               # 添加一类现金资产，净值为1
    data['month'] = data.index.map(lambda x: int(x / 100))
    months = data['month'].unique()
    monthly_dates = [data[data['month'] == m].index.tolist()[-1] for m in months]   # 每月最后一个交易日
    monthly_data = data_close.loc[monthly_dates, :]                                 # 每月月末收盘价
    monthly_return_data = (monthly_data / monthly_data.shift(1) - 1).iloc[1:]       # 每月月末计的涨跌幅
    daily_dates = data_close.index.tolist()                                         # 全部交易日
    momentum_data = pd.DataFrame(index=monthly_dates)                               # 动量数据
    momentum_data[3145] = monthly_data[3145] / monthly_data[3145].shift(6)
    momentum_data[3159] = monthly_data[3159] / monthly_data[3159].shift(3)
    momentum_data[4978] = monthly_data[4978] / monthly_data[4978].shift(3)
    momentum_data[6455] = monthly_data[6455] / monthly_data[6455].shift(1)
    momentum_data[14599] = monthly_data[14599] / monthly_data[14599].shift(1)

    rs_data = pd.DataFrame(index=daily_dates)                                       # RSRS数据
    for i in range(len(assets)):

        def f(x):
            tmp = pd.DataFrame(x)
            tmp['constant'] = 1
            res = sm.OLS(x['HighPrice'], x[['constant', 'LowPrice']]).fit()
            r2 = res.rsquared
            beta = res.params[-1]
            return {'r2': r2, 'beta': beta}

        rs_data[assets[i]] = data_list[i].rolling(18).apply(f)

    bt.GlobalParam.daily_dates = daily_dates
    bt.GlobalParam.monthly_dates = monthly_dates
    bt.GlobalParam.daily_close = data_close
    bt.GlobalParam.monthly_return_data = monthly_return_data
    bt.GlobalParam.momentum_data = momentum_data
    bt.GlobalParam.rs_data = rs_data
    bt.GlobalParam.asset_pool = [3145, 3159, 4978, 6455, 14599, 0]


def main():
    init_data()

    # init back test params
    bt.BktestParam.signal_params = {'risk': 0.03}
    bt.BktestParam.momentum_params = {}
    bt.BktestParam.rs_params = {}

    # print(data_close)
    bt.cal_long_weight()
    bt.cal_nav()
    pd.set_option('max_rows', 1000)
    print(bt.BktestResult.w)
    print(bt.BktestResult.nav)

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False      # 解决保存图像是负号'-'显示为方块的问题
    bt.GlobalParam.daily_close['test'] = bt.BktestResult.nav
    bt.GlobalParam.daily_close.plot(title='年化风险水平11%')
    plt.show()


if __name__ == '__main__':
    main()
