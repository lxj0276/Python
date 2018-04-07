## 金融和经济数据应用

### 数据规整化
**频率不同的时间序列的运算**
+ `resample` 将数据转换到固定频率，适合规整的索引
+ `reindex` 使数据符合一个新索引，适合不规整的索引
```py
ts1.resample('B', method='ffill')
# 规整数据
ts1.reindex(ts2.index, method='ffill')
# ts2的索引不规整
```

**如果索引是 `period` 时间数列**
此时和 `timestamp` 的时间序列不同，`Period` 索引 **必须** 进行显示转换。
+ `infl.asfreq('Q-SEP', how='end')`
+ 先将年化的infl索引转换为变成季度数据
+ `infl.reindex(gdp.index, method='ffill')`
+ 在使用重索引，将两个索引规整化

**选取特定时间点**
+ `ts[time(10,0)]`
+ 传入time对象就可以抽取时间点上的值
+ `ts.at_time((time(10,0)))`
+ 实际上是使用 `at_time` 方法
+ 如果没有刚好落在时间点上的数据，就只能取最近的数据
+ `selection = pd.date_range('', periods=4, freq='B')`
+ `ts.asof(selection)`
+ 传入的参数是要选取的日期范围和时间点
+ 得到这些时间点或**之前最近**的**有效非NA**数据

**拼接多个数据源**
+ `combine_first` 结合两个数据源取出非NA值
+ `spliced.update(data2, overwrite=False)` 可以得出相同结果
+ `overwrite=False` 只填补空白NA值
+ `concat` 拼接

### 金融时间序列处理
**有用的时间序列函数**
+ `series.pct_change()` 百分比变化。加一则为收益
+ `series.cumprod()` 累计积，用于计算**累计收益**
+ `series.cumsum()` 累计和
+ `series.diff()` **一阶差分**

**分组变化分析**
+ `zscore = lambda x:(x-x.mean())/x.std()`
+ `by_industry.apply(zscore)`
+ 分组分析

**分组因子暴露**
+ `factors.corwith(port)`
+ `pd.ols(y=port, x=factors).beta`
+ 用最小二乘回归计算因子暴露 
