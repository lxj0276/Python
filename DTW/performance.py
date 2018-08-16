import pandas as pd
from strategy import OPT


if __name__ == '__main__':
    name = "{}{}{}".format(OPT['trend_win'], OPT['return_win'], len(OPT['pos']))
    data = pd.read_csv(OPT['data_dir'], engine='python', index_col=0)
    nav1 = pd.read_csv('output/nav_{}.csv'.format(name), header=None, index_col=0)
    nav2 = pd.read_csv('output/nav_my_{}.csv'.format(name), header=None, index_col=0)

    # 基准Hs300
    base = pd.read_csv('data/3145.csv', index_col=0)
    base = base.reindex(nav1.index)['close']
    base = base / base[0]

    # hs300, 策略1, 策略2
    df = pd.DataFrame(columns=['hs300', '策略1', '策略2'])
    df.iloc[:, 0] = base
    df.iloc[:, 1] = nav1[1]
    df.iloc[:, 2] = nav2[1]
    df.index = pd.to_datetime(df.index)

    year = df.index.year
    year_starts = year.to_series().diff()
    year_starts = df.index[year_starts.fillna(1.0) > 0]

    df = df.loc[year_starts]
    df = df.shift(-1) / df
    df['策略1-超额收益'] = df['策略1'] - df['hs300']
    df['策略2-超额收益'] = df['策略2'] - df['hs300']
    df.dropna().to_csv('output/超额收益{}.csv'.format(name), encoding='gbk', float_format='%.3f')