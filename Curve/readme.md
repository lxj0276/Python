# 需求
对于股票日走势，标注出高点和低点，即拐点

## 实现方式
首先需要平滑化的曲线，消除掉噪声，使得曲线不受其他方面影响，这个可以使用 `savgol_filter` 进行滤波降噪。

```py
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except (ValueError, msg):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')
```

然后取高点和低点实际上就是取极值点。只要对滤波后的曲线取一阶差分，就能够得到趋势序列。
```py
curved_sery = pd.Series(arr, index=sery.index)
diff = curved_sery.diff() > 0
diff1 = diff.shift().fillna(method='backfill')
points_low = [diff[i] and (not diff1[i]) for i in range(len(diff))]
points_high = [not(diff[i]) and diff1[i] for i in range(len(diff))]
```

## python
+ `series` 和 `dataframe` 可以直接切片取行索引

## pandas
**升采样与插值**
```py
new_sery = pd.concat([sery[points_low], sery[points_high]]).sort_index()
new_sery = new_sery.reindex(sery.index).interpolate('linear')
```

**删除某一行或列**
+ `df.drop(labels, axis=1)`
可以删除若干行，若干列，指定 `inplace` 参数说明是就地修改还是生成新的数据框

## matplotlib
标注方法

```py
plt.plot(sery)
plt.plot(curved_sery)
# plt.plot(sery)
plt.plot(sery[points_low], 'ks')
plt.plot(sery[points_high], 'gs')
plt.show()
```
