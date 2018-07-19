import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def get_trend(series, window):
    """
    获取趋势
    :param series: 时间序列
    :param window: 窗口大小
    :return:
    """
    arr = np.array(series)
    arr = savgol_filter(arr, window, 5)                                     # 滤波
    curved_series = pd.Series(arr, index=series.index)
    diff = curved_series.diff() > 0                                         # 求差分
    diff1 = diff.shift().fillna(method='backfill')
    points_low = [diff[i] and (not diff1[i]) for i in range(len(diff))]     # 取出极小值点
    points_high = [not (diff[i]) and diff1[i] for i in range(len(diff))]    # 取出极大值点

    def rolling_arg(sry, win, method):
        n = len(sry)
        half = int(win / 2)
        res = pd.Series(sry.index, index=sry.index)
        for i in range(n):
            start = max(i - half, 0)
            end = min(i + half, n)
            if method == 'max':
                res[i] = sry[start: end].idxmax()
            else:
                res[i] = sry[start: end].idxmin()
        return res

    # 对滚动窗口求 argmax 和 argmin
    rolling_max = rolling_arg(series, window/2, method='max')
    rolling_min = rolling_arg(series, window/2, method='min')

    # 以滤波曲线的极大极小值为锚点，选出原曲线中离锚点最近的高点和低点
    dates_high = rolling_max[points_high].drop_duplicates()
    dates_low = rolling_min[points_low].drop_duplicates()

    # 生成初步结果
    up = pd.Series(np.ones(len(dates_high)), index=dates_high)
    down = pd.Series(np.ones(len(dates_low)) * -1, index=dates_low)
    res = pd.concat([up, down]).sort_index()

    # 整理结果，对连续的高点，插入低点，连续的低点则插入高点
    dates = res.index.tolist()
    for i in range(len(res) - 1):
        if res[i] + res[i + 1] == -2:
            date = series[dates[i]: dates[i + 1]].idxmax()
            res[date] = 1
        elif res[i] + res[i + 1] == 2:
            date = series[dates[i]: dates[i + 1]].idxmin()
            res[date] = -1
        else:
            pass

    res = res.sort_index()

    # 整理结果
    res = pd.DataFrame(res)
    res['date'] = res.index
    res = pd.concat([res, res.shift(-1)], axis=1).dropna()
    res.columns = [0, 'start', 1, 'end']
    res['sign'] = (res[1] - res[0]) / 2
    res = res[res['sign'] != 0.0]

    # 过滤收益率变化小与 10% 的片段
    def func(x, s):
        return s[x['end']] / s[x['start']] - 1

    r = res.apply(lambda x: func(x, series), axis=1)
    res = res[r * res['sign'] > 0.1]
    dates = res.index.tolist()

    # 将过滤后连续下降或上涨的片段整合
    mark = np.ones(len(res))
    for i in range(len(res) - 1):
        if res.loc[dates[i], 'sign'] + res.loc[dates[i+1], 'sign'] != 0:
            mark[i+1] = 0.0
    res = res[mark != 0.0]

    dates = res.index.tolist()
    for i in range(len(res) - 1):
        res.loc[dates[i], 'end'] = res.loc[dates[i+1], 'start']

    return res[['start', 'end', 'sign']]


def plot_trend(series, result):
    points_low = result[result['sign'] == -1]['end']
    points_high = result[result['sign'] == 1]['end']

    plt.plot(series)
    plt.plot(series[points_low], 'ks')
    plt.plot(series[points_high], 'gs')
    plt.show()


if __name__ == '__main__':
    # 获取数据
    print('get data:')
    data = ts.get_hist_data('600848')
    data = data['ma5']
    data.index = pd.to_datetime(data.index)
    series = data[::-1]
    print('compute:')
    result = get_trend(series, 41)
    print(result)
    plot_trend(series, result)


