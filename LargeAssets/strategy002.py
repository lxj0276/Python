import pandas as pd
import numpy as np
import FactorModel as Fm
import backtest as bt
import matplotlib.pyplot as plt
from pylab import mpl


# init back test params
@bt.signal
def func(date, risk):
    daily_dates = bt.GlobalParam.daily_dates
    now = daily_dates.index(date)
    if now < 12:
        return False, None

    now_data = bt.GlobalParam.data.iloc[now-12:now, :].dropna(axis=1)
    predicts = now_data.ewm(alpha=0.8).mean().iloc[-1]
    rs = np.cov(np.asmatrix(now_data).T)

    r = (1 + risk) ** (1 / 12) - 1
    up = predicts.iloc[-1] / r + 0.1
    up = up if up < 1.0 else 1.0
    try:
        w = Fm.min_risk(returns=predicts, risk_structure=rs, target_return=r, up=up)
        w = pd.Series(w, index=now_data.columns)
        w = w.reindex(bt.GlobalParam.data.columns)
        return round(np.nansum(w), 1) == 1.0, np.asarray(w)
    except ValueError:
        return False, None


if __name__ == '__main__':
    # init data
    data = pd.read_csv('data/IndexPrice.csv', index_col='DATES')
    data.columns = data.columns.map(str.lower)
    data['month'] = data.index.map(lambda x: int(x / 100))
    tickers = data.groupby(data['ticker'])

    df = pd.DataFrame(index=data['month'].unique())
    for name, t in tickers:
        df[name] = t['close'].groupby(t['month']).apply(lambda x: x.iloc[-1])

    index = pd.period_range(start='2005/01', end='2018/05', freq='M')
    df.index = index
    data_close = df.apply(lambda x: x / x.dropna()[0])
    df = (df / df.shift(1) - 1).iloc[1:]

    # init global params
    bt.GlobalParam.data = df
    bt.GlobalParam.daily_close = data_close
    bt.GlobalParam.daily_dates = data_close.index.tolist()
    bt.GlobalParam.asset_pool = ['3145', '3159', '4978', '6455', '14599']

    # init back test params
    bt.BktestParam.signal_params = {'risk': 0.03}

    # print(data_close)
    refresh_dates, w = bt.cal_long_weight()
    bt.BktestParam.refresh_dates = refresh_dates
    bt.BktestResult.w = w
    bt.BktestResult.nav = bt.cal_nav()
    pd.set_option('max_rows', 1000)
    print(w)
    print(bt.BktestResult.nav)

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    data_close['test'] = bt.BktestResult.nav
    data_close.plot(title='年化风险水平11%')
    plt.show()