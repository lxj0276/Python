## 时间序列

**日期和时间数据类型及工具**
```py
from datetime import datetime
now = datetime.now()
delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
delta.days # 可以取出相隔的天数
delta.seconds # 可以取出相隔的秒数
```

**字符串和 `datetime` 的相互转换**
+ `stamp.strftime('%Y-%m-%d')` 输出
+ `datetime.strptime(value, '%Y-%m-%d')` 从字符串 `value` 中读入
+ `pd.to_datetime(datestr)` 利用 `pandas` 自带的模块

### 时间序列基础
**索引，切片**
+ `ts['1/10/2011']` 利用解释为日期的字符串选取
+ `longer['2001']` 较长的数据传入年轻松完成**切片**
+ `longer['2001-05']` 年月完成切片

**日期的范围、频率和移动**
+ `ts.resample('D')` 按照天重新设定频率
+ `pandas.date_range('', '')` 生成指定长度的索引
+ `ts.shift(3, freq='D')`
```py
# 默认freq参数是D
pd.date_range('1/1/2000', '12/1/2000', freq='BM')
# BM表示 business end of month，设定了采样频率
freq = 'WOM-3FRI'
# 每月的第三个周三

ts.shift(3, freq='D')
# 不指定freq则索引不变，数据移动
# 指定则索引变化，数据不变
```
**锚点偏移量**
+ `MothEnd(2)` 之后的第二个月末
+ `MonthEnd().rollforward(now)`
偏移量可以**加在** `datetime` 对象上，加的是锚点偏移量，就会是原日期向前或向后滚动到下一个日期。
```py
now + MonthEnd(2)
# 之后的第二个月末

offset = MonthEnd()
offset.rollforward(now) # 向前滚动
offset.rollback(now) # 向后滚动

```

### 时期及其算术运算
用 `period` 表示时间区间，如某个月份，某个季度，并且可以进行算数运算。
+ `p.asfreq('M', how='start')` 低转高
+ 高转低时，由子时期所属的位置决定
+ `ts.asfreq()` 可以对ts使用
```py
pd.Period(2007, freq='A-DEC')
# 一个Period对象

rng = pd.period_range('1/1/2000', '6/30/2000', freq='M')
# 一组Period，可以用作索引

ts.asfreq('M', how='start')
#　改变series中的频率
```

**降采样、升采样**
+ 将高频转换为低频，并用某种方式**聚合**
+ `ts.resample('M', how='mean')`
+ `ts.resample('5min', how='ohlc')`
+ 这种重采样聚合方式可以得到**开盘最高最低收盘值**
+ ts.groupby(lambda x:x.month).mean()
+ 也可以利用groupby
+ ts.resample('D', fill_method='ffill')
+ 将低频转为高频叫做升采样， 插值方法和 `fillna` `reindex` 一样

### 移动窗口函数
移动窗口自动排除缺失值。
+ `pd.rolling_mean(series, 250, min_periods=10)`
+ 参数中指定了必须有10个非NA值，这是在求**250日均线**
+ `rolling_count` 非NA观测值
+ `rolling_sum` 移动窗口和
+ `rolling_meadian` 中位数
+ `rolling_apply` 对窗口应用普通数组函数
+ `ewma` 指数加权平均 **赋予近期更大的权数**
+ ...
+ **二元移动窗口函数**
+ `pd.rolling_corr(r1, r2, 125, min_periods=100)`
+ 可以用来计算与标普500的移动相关系数
