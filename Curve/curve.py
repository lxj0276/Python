import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def get_trend(series):
    arr = np.array(series)
    arr = savgol_filter(arr, 11, 3)
    curved_series = pd.Series(arr, index=series.index)
    diff = curved_series.diff() > 0
    diff1 = diff.shift().fillna(method='backfill')
    points_low = [diff[i] and (not diff1[i]) for i in range(len(diff))]
    points_high = [not (diff[i]) and diff1[i] for i in range(len(diff))]

    sign = np.ones(points_low.count(True))
    up_period = pd.DataFrame({'start': series[points_low].index,
                              'end': series[points_high].index,
                              'sign': sign})
    down_period = pd.DataFrame({'end': series[points_low].index[1:],
                                'start': series[points_high].index[:-1],
                                'sign': sign[:-1] * -1})

    result = pd.concat([up_period, down_period])
    return result


def plot_trend(series):
    arr = np.array(series)
    arr = savgol_filter(arr, 11, 3)
    curved_series = pd.Series(arr, index=series.index)
    diff = curved_series.diff() > 0
    diff1 = diff.shift().fillna(method='backfill')
    points_low = [diff[i] and (not diff1[i]) for i in range(len(diff))]
    points_high = [not (diff[i]) and diff1[i] for i in range(len(diff))]

    plt.plot(series)
    plt.plot(curved_series)
    # plt.plot(series)
    plt.plot(series[points_low], 'ks')
    plt.plot(series[points_high], 'gs')
    plt.show()


if __name__ == '__main__':
    # 获取数据
    data = ts.get_hist_data('600848')
    data = data['ma5']
    data.index = pd.to_datetime(data.index)
    series = data['2017-01-01'::-1]
    print(get_trend(series))
    plot_trend(series)

