import pandas as pd
from strategy import OPT


def win_rate(base, nav, freq='d'):
    if type(base.index) != pd.DatetimeIndex:
        base.index = pd.to_datetime(base.index)
    if type(nav.index) != pd.DatetimeIndex:
        nav.index = pd.to_datetime(nav.index)

    nav = nav.reindex(base.index)
    if freq == 'm':
        months = base.index.year * 100 + base.index.month
        starts = months.to_series().diff().fillna(1) > 0
        nav = nav[starts.tolist()]
        base = base[starts.tolist()]
    elif freq == 'd':
        pass
    else:
        raise Exception("没有定义的方法：{}".format(freq))

    nav_return = (nav.shift(-1) / nav).dropna()
    base_return = (base.shift(-1) / base).dropna()
    win = len(nav_return[nav_return > base_return])
    total = len(nav_return)

    return win / total


def performance(base, nav, name):
    if type(base.index) != pd.DatetimeIndex:
        base.index = pd.to_datetime(base.index)
    if type(nav.index) != pd.DatetimeIndex:
        nav.index = pd.to_datetime(nav.index)

    # hs300, 策略
    df = pd.DataFrame(columns=[name[0], name[1]])
    df.iloc[:, 0] = base
    df.iloc[:, 1] = nav
    df.index = pd.to_datetime(df.index)

    year = df.index.year
    year_starts = year.to_series().diff()
    year_starts = df.index[year_starts.fillna(1.0) > 0]
    year_starts = year_starts.insert(len(df.index), df.index[-1])

    # 超额收益
    df = df.loc[year_starts]
    df = df.shift(-1) / df - 1
    df['{}-超额收益'.format(name[1])] = df[name[1]] - df[name[0]]
    # 胜率
    for i in range(len(df.index)-1):
        begin, end = df.index[i], df.index[i+1]
        df.loc[begin, '日胜率'] = win_rate(base[begin:end], nav[begin:end], 'd')
        df.loc[begin, '月胜率'] = win_rate(base[begin:end], nav[begin:end], 'm')

    summary = [base[-1] / base[0] - 1, nav[-1] / nav[0] - 1, 0, 0, 0]
    summary[2] = summary[1] - summary[0]
    summary[3] = win_rate(base, nav, 'd')
    summary[4] = win_rate(base, nav, 'm')
    df = df.iloc[:-1, :]
    df.index = df.index.map(lambda x: str(x.year))
    df.loc['汇总', :] = summary

    df = df.applymap(lambda x: '{:.2%}'.format(x))
    return df


def main():
    nav = pd.read_csv('output/zz800_s1.csv', header=None, index_col=0)

    # 基准Hs300
    base = pd.read_csv(OPT['base_dir'], index_col=0)
    base = base.reindex(nav.index)['close']
    base = base / base[0]

    df = performance(base, nav[1], ['hs300', '中证策略'])

    df.to_csv('output/performance-zz.csv', encoding='gbk')


if __name__ == '__main__':
    main()