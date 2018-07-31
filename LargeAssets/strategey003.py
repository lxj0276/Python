import pandas as pd
import numpy as np
import FactorModel as Fm
from Back import Context
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pylab import mpl


# init back test params
def func(context, date, risk):
    monthly_dates = context.GlobalParam['monthly_dates']
    if date in monthly_dates:
        now = monthly_dates.index(date)
        if now < 12:
            return False, None

        now_data = context.GlobalParam['monthly_return_data'].iloc[now - 12:now, :].dropna(axis=1)
        predicts = now_data.ewm(alpha=0.8).mean().iloc[-1]
        rs = np.cov(np.asmatrix(now_data).T)

        r = (1 + risk) ** (1 / 12) - 1
        up = predicts.iloc[-1] / r + 0.1
        up = up if up < 1.0 else 1.0
        try:
            # 风险平价
            w = Fm.min_risk(returns=predicts, risk_structure=rs, target_return=r, up=up)
            w = pd.Series(w, index=now_data.columns)
            w = w.reindex(context.GlobalParam['asset_pool']).fillna(0)
            if round(np.nansum(w), 1) != 1.0:
                return False, None

            # 动量因子
            w = factor_adj(context, w, date, **context.BktestParam['momentum_params'])
            # RSRS
            w = rs_adj(context, w, date, **context.BktestParam['rs_params'])
            return True, np.asarray(w)
        except ValueError:
            return False, None
    else:
        if len(context.BktestResult['w']) == 0:
            return False, None
        old = context.BktestResult['w'].iloc[-1, :]
        new = rs_adj(context, old, date, **context.BktestParam['rs_params'])
        return np.nansum(np.abs(old - new)) == 0, new


def func2(context, date, risk):
    monthly_dates = context.GlobalParam['monthly_dates']
    if not (date in monthly_dates):
        return False, None

    now = monthly_dates.index(date)
    if now < 12:
        return False, None

    now_data = context.GlobalParam['monthly_return_data'].iloc[now-12:now, :].dropna(axis=1)
    predicts = now_data.ewm(alpha=0.8).mean().iloc[-1]
    rs = np.cov(np.asmatrix(now_data).T)

    r = (1 + risk) ** (1 / 12) - 1
    up = predicts.iloc[-1] / r + 0.1
    up = up if up < 1.0 else 1.0
    try:
        w = Fm.min_risk(returns=predicts, risk_structure=rs, target_return=r, up=up)
        w = pd.Series(w, index=now_data.columns)
        w = w.reindex(context.GlobalParam['asset_pool'])
        return round(np.nansum(np.abs(w)), 1) == 1.0, np.asarray(w)
    except ValueError:
        return False, None


def func3(context, date, risk):
    monthly_dates = context.GlobalParam['monthly_dates']
    if date in monthly_dates:
        now = monthly_dates.index(date)
        if now < 12:
            return False, None

        now_data = context.GlobalParam['monthly_return_data'].iloc[now - 12:now, :].dropna(axis=1)
        predicts = now_data.ewm(alpha=0.8).mean().iloc[-1]
        rs = np.cov(np.asmatrix(now_data).T)

        r = (1 + risk) ** (1 / 12) - 1
        up = predicts.iloc[-1] / r + 0.1
        up = up if up < 1.0 else 1.0
        try:
            # 风险平价
            w = Fm.min_risk(returns=predicts, risk_structure=rs, target_return=r, up=up)
            w = pd.Series(w, index=now_data.columns)
            w = w.reindex(context.GlobalParam['asset_pool']).fillna(0)
            if round(np.nansum(w), 1) != 1.0:
                return False, None

            # 动量因子
            w = factor_adj(context, w, date, **context.BktestParam['momentum_params'])
            return True, np.asarray(w)
        except ValueError:
            return False, None
    else:
        return False, None


def factor_adj(context, w, date, k):
    """
    依赖于经过标准化后的
    数据1：每列对应每个资产使用的动量因子
    数据2：每列对应每个资产使用的估值因子 【见对照表】
    对每一类资产：
        取出当期的动量因子值，与过去一年【12个月】平均值比较，大于则权重更新乘以 K， 否则更新为 1/K
        取出当期的估值因子值，与2005年至当期所有已知因子的均值比较，与上文采取同样的调整方式
    计算完毕，将权重归一化
    """
    monthly_dates = context.GlobalParam['monthly_dates']
    momentum_data = context.GlobalParam['momentum_data']
    now = monthly_dates.index(date)
    sig = momentum_data.iloc[now, :] - momentum_data.iloc[now-12:now, :].mean()

    low = sig <= 0
    high = sig > 0
    sig[low] = 1 / k
    sig[high] = k

    sig = sig.reindex(context.GlobalParam['asset_pool'])
    sig = sig * w
    w = (sig * w[:-1].sum() / sig.sum()).fillna(w.iloc[-1])
    return w


def rs_adj(context, w, date, m, s):
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
    daily_dates = context.GlobalParam['daily_dates']
    rs_data = context.GlobalParam['rs_data']
    now = daily_dates.index(date)

    sig = pd.Series(index=context.GlobalParam['asset_pool'][:-1])
    for i in sig.index:
        arr = rs_data[i].iloc[max(now - m, 0):now + 1, 0]
        z = (arr - arr.mean()) / arr.std()
        sig[i] = rs_data[i].loc[date, 'r2'] * z.iloc[-1]

    sig = sig - s
    sig = sig.reindex(context.GlobalParam['asset_pool']).fillna(1)
    w[sig < 0] = 0
    w.iloc[-1] = 1 - w[:-1].sum()

    return w


def init_data(context):
    assets = [3145, 3159, 4978, 6455, 14599]

    data_list = [pd.read_csv('data/{}.csv'.format(i), index_col='Date')
                 for i in assets]

    data = pd.DataFrame(index=data_list[0].index)
    for i in range(len(assets)):
        data[assets[i]] = np.asarray(data_list[i]['close'])

    data_close = data.apply(lambda x: x / (x.dropna().iloc[0]))                            # 各类资产净值曲线

    data_close[0] = 1                                                               # 添加一类现金资产，净值为1
    data['month'] = data.index.map(lambda x: "".join(x.split('-')[:-1]))
    months = data['month'].unique()
    monthly_dates = [data[data['month'] == m].index.tolist()[-1] for m in months]   # 每月最后一个交易日
    monthly_data = data_close.loc[monthly_dates, assets]                            # 每月月末收盘价
    monthly_return_data = (monthly_data / monthly_data.shift(1) - 1).iloc[1:]       # 每月月末计的涨跌幅
    daily_dates = data_close.index.tolist()                                         # 全部交易日
    momentum_data = pd.DataFrame(index=monthly_dates)                               # 动量数据
    momentum_data[3145] = monthly_data[3145] / monthly_data[3145].shift(6)
    momentum_data[3159] = monthly_data[3159] / monthly_data[3159].shift(3)
    momentum_data[4978] = monthly_data[4978] / monthly_data[4978].shift(3)
    momentum_data[6455] = monthly_data[6455] / monthly_data[6455].shift(1)
    momentum_data[14599] = monthly_data[14599] / monthly_data[14599].shift(1)

    # RSRS数据
    # rs_data = pd.DataFrame(index=daily_dates,
    #                        columns=pd.MultiIndex.from_product([assets, ['r2', 'beta']]))
    #
    # def f(x):
    #     tmp = pd.DataFrame(x)
    #     tmp['constant'] = 1
    #     try:
    #         res = sm.OLS(tmp['high'], tmp[['constant', 'low']], missing='drop').fit()
    #         r2 = res.rsquared
    #         beta = res.params[-1]
    #         return {'r2': r2, 'beta': beta}
    #     except ValueError:
    #         return {'r2': np.nan, 'beta': np.nan}
    #
    # for j in range(len(daily_dates)):
    #     line = []
    #     for i in range(len(assets)):
    #         series = {'r2': np.nan, 'beta': np.nan}
    #         if j > 17:
    #             series = f(data_list[i].iloc[j-18:j, :]) if j > 17 else series
    #         line.extend(series.values())
    #
    #     rs_data.iloc[j, :] = line
    #
    # rs_data.to_csv('data/rs_data.csv')
    # RSRS数据

    rs_data = pd.read_csv('data/rs_data.csv', header=[0, 1], index_col=0)
    rs_data.columns = pd.MultiIndex.from_product([assets, ['r2', 'beta']])
    context.GlobalParam['daily_dates'] = daily_dates
    context.GlobalParam['monthly_dates'] = monthly_dates
    context.GlobalParam['daily_close'] = data_close
    context.GlobalParam['monthly_return_data'] = monthly_return_data
    context.GlobalParam['momentum_data'] = momentum_data
    context.GlobalParam['rs_data'] = rs_data
    context.GlobalParam['asset_pool'] = [3145, 3159, 4978, 6455, 14599, 0]


def back_test(context, risk):
    # init back test params
    context.BktestParam['signal_params'] = {'risk': risk}
    context.BktestParam['momentum_params'] = {'k': 2}
    context.BktestParam['rs_params'] = {'m': 600,
                                        's': [0.7, 0.5, 0.8, 0, 0]}

    context.BktestParam['signal'] = func2
    w0 = context.cal_long_weight()
    strategy0 = context.cal_nav()

    context.BktestParam['signal'] = func3
    w1 = context.cal_long_weight()
    strategy1 = context.cal_nav()

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False      # 解决保存图像是负号'-'显示为方块的问题
    context.GlobalParam['daily_close']['风险平价'] = strategy0
    context.GlobalParam['daily_close']['风险平价+动量'] = strategy1
    context.GlobalParam['daily_close'].plot(title='年化风险水平{:.1f}%'.format(risk * 100))
    plt.savefig('{:.1f}%.png'.format(risk * 100))

    return w0, w1


def main():
    context = Context()
    # init global params
    init_data(context)

    back_test(context, 0.2)
    # writer1 = pd.ExcelWriter('output/风险平价.xlsx')
    # writer2 = pd.ExcelWriter('output/风险平价+动量.xlsx')
    # for r in [0.03, 0.05, 0.07, 0.09]:
    #     print('running:{}'.format(r))
    #     w0, w1 = back_test(context, r)
    #     w0.to_excel(writer1, str(r))
    #     w1.to_excel(writer2, str(r))
    #
    # writer1.save()
    # writer2.save()


if __name__ == '__main__':
    main()
