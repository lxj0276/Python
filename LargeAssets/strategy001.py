import pandas as pd
import numpy as np

import optimus as op
from back import Context
from config import W_DIR, PIC_DIR

import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False      # 解决保存图像是负号'-'显示为方块的问题


# init back test params


def signal(context, date, risk):
    monthly_dates = context.Add['monthly_dates']

    if date in monthly_dates:
        now = monthly_dates.index(date)
        if now < 12:
            return False, None

        now_data = context.Add['monthly_return_data'].iloc[now - 12:now, :].dropna(axis=1)
        predicts = now_data.ewm(alpha=0.8).mean().iloc[-1]
        rs = np.cov(np.asmatrix(now_data).T)

        r = (1 + risk) ** (1 / 12) - 1
        up = predicts.iloc[-1] / r + 0.1
        up = up if up < 1.0 else 1.0
        try:
            # 风险平价
            w = op.min_risk(returns=predicts, risk_structure=rs, target_return=r, up=up)
            w = pd.Series(w, index=now_data.columns)
            w = w.reindex(context.GlobalParam['asset_pool']).fillna(0)
            if round(np.nansum(w), 1) != 1.0:
                return False, None

            # 动量因子
            w = factor_adj(context, w, date, **context.Add['momentum_params'])
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
    monthly_dates = context.Add['monthly_dates']
    momentum_data = context.Add['momentum_data']
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


def init_data(context):
    assets = [3145, 3159, 4978, 6455, 14599]

    data_list = [pd.read_csv('data/{}.csv'.format(i), index_col='Date')
                 for i in assets]

    data = pd.DataFrame(index=data_list[0].index)
    for i in range(len(assets)):
        data[assets[i]] = np.asarray(data_list[i]['close'])

    # 净值数据
    data_close = data.apply(lambda x: x / (x.dropna().iloc[0]))                     # 各类资产净值曲线
    data_close[0] = 1                                                               # 添加一类现金资产，净值为1

    # 动量数据
    data['month'] = data.index.map(lambda x: "".join(x.split('-')[:-1]))
    months = data['month'].unique()
    monthly_dates = [data[data['month'] == m].index.tolist()[-1] for m in months]   # 每月最后一个交易日
    monthly_data = data_close.loc[monthly_dates, assets]                            # 每月月末收盘价
    monthly_return_data = (monthly_data / monthly_data.shift(1) - 1).iloc[1:]       # 每月月末计的涨跌幅
    momentum_data = pd.DataFrame(index=monthly_dates)
    momentum_data[3145] = monthly_data[3145] / monthly_data[3145].shift(6)
    momentum_data[3159] = monthly_data[3159] / monthly_data[3159].shift(3)
    momentum_data[4978] = monthly_data[4978] / monthly_data[4978].shift(3)
    momentum_data[6455] = monthly_data[6455] / monthly_data[6455].shift(1)
    momentum_data[14599] = monthly_data[14599] / monthly_data[14599].shift(1)

    context.Add['monthly_dates'] = monthly_dates
    context.Add['momentum_data'] = momentum_data
    context.Add['monthly_return_data'] = monthly_return_data
    context.GlobalParam['daily_close'] = data_close

    return data_close


def back_test(context, risk):
    # init back test params
    context.Add['signal_params'] = {'risk': risk}
    context.Add['momentum_params'] = {'k': 2}

    context.BktestParam['signal'] = signal
    w = context.cal_long_weight()
    nav = context.cal_nav()
    return w, nav


def main():
    context = Context()
    # init global params
    data_close = init_data(context)

    writer = pd.ExcelWriter(W_DIR + '风险平价+动量.xlsx')
    for r in [0.03, 0.05, 0.07, 0.09, 0.2]:
        print('running:{}'.format(r))
        w, nav = back_test(context, r)
        w.to_excel(writer, str(r))

        data_close['风险平价+动量'] = nav
        data_close.index = pd.to_datetime(data_close.index)
        data_close.plot(title='目标收益水平{:.1f}%'.format(r * 100))
        plt.savefig(PIC_DIR + '{:.1f}%.png'.format(r * 100))

    writer.save()


if __name__ == '__main__':
    main()
