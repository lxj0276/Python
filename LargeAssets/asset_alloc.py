import pandas as pd
import numpy as np
from FactorModel import min_risk, max_returns


def predict_month_return(data):
    data.resample('M').apply()


def predict(data):
    return data.ewm(alpha=0.8).mean().iloc[-1]


def allocate(data, method, arg):
    """
    配置资产
    :param data: 年度数据
    :param method: 配置方法，最小化风险或是最大化收益
    :param arg: 额外参数，最小化风险时为目标收益，最大化收益时为风险
    :param up: 大类上限
    :return: 返回配置结果
    """
    predicts = predict(data)
    rs = np.cov(np.asmatrix(data).T)

    if method == 'min_risk':
        r = (1 + arg) ** (1 / 12) - 1
        up = predicts.iloc[-1] / r + 0.1
        up = up if up < 1.0 else 1.0
        w = min_risk(returns=predicts, risk_structure=rs, target_return=r, up=up)
        print(up)
        print(w)
        risk = (np.dot(w.T, np.dot(rs, w))[0, 0]) ** 0.5 * (12 ** 0.5)  # 年化
        return w, risk
    else:
        w = max_returns(returns=predicts, risk_structure=rs, risk=arg)
        r = np.dot(w.T, np.asarray(predicts))[0]
        return w, r


def back_test(df, num, method, arg):
    """
    回测
    :param df: 数据
    :param num: 每期回溯期数
    :param method: 最大化收益 or 最小化风险
    :param arg: 所需要的额外参数
    :return: 返回回测数据
    """
    cols = df.columns.tolist()
    cols.append('target')
    res = pd.DataFrame(columns=cols, index=df.index, dtype='float64')
    for i in range(num, len(df)+1):
        try:
            weights, target = allocate(df.iloc[i-num:i].dropna(axis=1), method, arg)
            for j in range(len(weights)):
                res.iloc[i-1, j] = weights[j]
                res.iloc[i-1, 5] = target
        except ValueError:
            print("not success: {}".format(i))
    return res[num-1:]


def max_drag_down(quot, trans, func):
    """
    计算最大回撤
    :param quot: DataFrame, 历史行情， 日涨跌幅
    :param trans: DataFrame, 每一期调仓的值
    :param func: 能够将历史行情与调仓的index匹配起来的函数
    :return: 最大回撤
    """
    def f(x, w):
        x.index = x['ticker']
        x = x.reindex(w[:-1].index).fillna(1.0)
        return np.dot(np.asarray(w[:-1].fillna(0)).T, np.asarray(x['pctchg']))

    s = pd.Series(np.nan, index=quot.index.unique())
    dates = quot.groupby(quot.index)
    for i, date in dates:
        try:
            s[i] = f(date, trans.loc[func(i)])
        except KeyError:
            pass

    s = s.dropna()
    s.groupby(s.index.map(lambda x: int(x/100)))


def main():
    data = pd.read_csv('data/IndexPrice.csv', index_col='DATES')
    data.columns = data.columns.map(str.lower)
    data['month'] = data.index.map(lambda x: int(x/100))
    tickers = data.groupby(data['ticker'])

    df = pd.DataFrame(index=data['month'].unique())
    for name, t in tickers:
        df[name] = t['close'].groupby(t['month']).apply(lambda x: x.iloc[-1])

    index = pd.period_range(start='2005/01', end='2018/05', freq='M')
    df.index = index
    df = (df / df.shift(1) - 1).iloc[1:]

    res = pd.DataFrame(columns=df.columns, index=df.index, dtype='float64')
    for i in range(12, len(df)):
        res.iloc[i] = predict(df.iloc[i-12:i].dropna(axis=1))

    print(res)
    res1 = back_test(df, 12, 'min_risk', 0.03)
    res2 = back_test(df, 12, 'min_risk', 0.05)
    res3 = back_test(df, 12, 'min_risk', 0.07)
    res4 = back_test(df, 12, 'min_risk', 0.09)

    res1.to_csv('output/0.03.csv', float_format='%.4f', date_format='%Y%m')
    res2.to_csv('output/0.05.csv', float_format='%.4f', date_format='%Y%m')
    res3.to_csv('output/0.07.csv', float_format='%.4f', date_format='%Y%m')
    res4.to_csv('output/0.09.csv', float_format='%.4f', date_format='%Y%m')

    res5 = back_test(df, 12, 'max_returns', 0.03)
    res6 = back_test(df, 12, 'max_returns', 0.05)
    res7 = back_test(df, 12, 'max_returns', 0.07)
    res8 = back_test(df, 12, 'max_returns', 0.09)
    res5.to_csv('output/0.03risk.csv', float_format='%.4f', date_format='%Y%m')
    res6.to_csv('output/0.05risk.csv', float_format='%.4f', date_format='%Y%m')
    res7.to_csv('output/0.07risk.csv', float_format='%.4f', date_format='%Y%m')
    res8.to_csv('output/0.09risk.csv', float_format='%.4f', date_format='%Y%m')


if __name__ == '__main__':
    main()
