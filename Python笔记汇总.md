# `Python` 笔记汇总

## `Python` 基础
+ `*` 符号的使用
```py
def make_group(num):
    index = [list(combinations(range(num), i)) for i in range(1, num+1)]
    return list(chain(*index))
```

+ `Warnings`
```py
import warnings
warnings.warn("")
```

+ `assert` 关键词，用来确信某一表达式是真的，在为假时报 `AssertionError`
```py
assert len(mylist) >= 1
```

+ `exec` 关键词
```py
exec 'print "Hello World"'
```

+ `import` 引入自己写的模块
```py
import sys;  
sys.path.append("路径名")  
import model
```

+ `a[::-1]` 倒序

+ `format()` 格式化输出
```py
print('{} is {:.2f}'.format(1.123,1.123)) # 取2位小数
print('{0} is {0:>10.2f}'.format(1.123))  # 取2位小数，右对齐，取10位
```

+ 去换行符
```py
line = line.strip('\n')
```

+ 字典默认值
对于普通的 `dict` 找不到键值会报错，有几种解决方案
`dict.setdefault(key, '')` 给key设定为默认值，如果key已经存在就不改变原来的值
`dict.get(key,'')` 在获取键key时，如果不存在key就设置新key，并且设置默认值，在获取 **未知存在** 的键时非常有用
`collections.defaultdict(int)` 参数给定的是类型，将返回类型默认值
也可以传入一个 **无参函数规定默认值**

+ 正则表达式
```py
pattern = re.compile('\\[(.*)\\]')
pattern.match(str(list(data.loc[i]))).group(1)
# \\在转义后成为\，在正则表达式里面表示转义斜杠
# group(1)取出匹配的组
```
关于字符串的优化问题：**熟练使用, `''.format()` 方法简化语法**
```py
string = '(' + pattern.match(str(list(data.loc[i]))).group(1) + ')'

# 简化为
string = '({0})'.format(pattern.match(str(list(data.loc[i]))).group(1))
```

## `numpy`

**numpy不支持字符串存储**

+ `matrix` 的索引方式和 `ndarray` 不同
```py
matrix[0, 0]
ndarray[0][0]
```

+ `np.dot` 可以用于矩阵乘法，但是只限于两个矩阵相乘

+ `np.cov` 默认是行与行之间求协方差，要求列的协方差需要 **先转置**

+ `np.asmatrix` 后有了 `T` 属性用于转置

+ `np.append` 用于往 `array` 里面添加元素， 记得 **往回赋值**

+ `np.fft` 傅里叶变换
`np.fft.fft`:
`np.fft.ifft`: 傅里叶逆变换
`ndarray.conjugate()`: 求共轭
`ndarray.real`: 取实数部分

+ `np.pad()` 数组补
`pad_width = (1, 1)`: 两边
`mode = 'constant'` 常数

+ `np.exp()` 指数运算

+ `np.column_stack((a, b))`
`np.row_stack((a, b))` 合并
`np.insert()`

+ `np.linalg.lstsq(x, y)[0]` 线性回归取系数

+ `np.sort` 排序
`np.sort(a)[::-1]` 倒序
`np.argsort()` 索引排序

+ `np.nansum()` 含缺失值的求和
`np.nancumsum()` 一系列含有 `nan` 求值ֵ

+ `np.reshape(-1)` 多维转一维，同样也可以一维转多维 **不产生复制**

+ `matlab` 中 `std` 为 `ny.std(ddof=1)` 要设置自由度

## `pandas`

+ `dropna(inplace=True)` 对目前的数据框进行修改，不产生新数据框

+ `series` 和 `dataframe` 可以直接切片取行索引

+ 选择过滤操作可以用 `bool` 索引实现

+ `astype` 可以转换数据类型，也可以在创建时指定 `dtype`
```py
dtype = 'str'
.astype('str')
```

+ 只有 `Series` 有 `asfreq()` 方法，可以将时间序列索引

+ 只有 `Series` 以及 `DataFrame` 有 `resample` 方法

+ `applymap` 会对每一个元素执行重复操作

### `Index`
+ `Index` 有 `append`,`delete`,`drop`,`insert` 以及一些集合方法

+ 利用pandas的算术功能时，**索引自动对齐** 会影响道计算结果，应该去除这个影响

+ 赋值时用 `Series` 对一个 `DataFrame`  赋值索引会对齐 **DataFrame** 的索引，即便长度一样也会产生NA。

+ `s1.index.union(s2)` 会使用 `s2` 的值和 `s1` 的索引求并集

+ `reindex` 后要赋值回去，**否则不会改变**

+ `reindex` 和 `index=` 的区别是前者是 **重采样** 而后者是 **更换**

+ `Index` 只有 `map` 方法没有 `apply` 方法

+ 多重索引有 `to_frame()` 方法方便转换为正常的 `DataFrame`

+ 多重索引有 `swaplevel([0, 1])` 方法可以转换 `level`

+ 多重索引有 `names` 属性

+ 可以直接多使用多重索引的数据框使用 `swaplevel`，这样可以用 `.loc[]` 选取想要的维度

+ `MultiIndex.from_arrays()` 还有很多类似其他函数可以用于创建多重索引

**时间序列索引**
+ `pd.to_datetime()` 将一个字符串序列转换为 `datetime` 序列

+ `dateutil.parser.parse` 将普通字符串转化巍峨 `datetime`

+ 可以用 `astype('str')` 来进行字符串的转换，再用 `pd.to_datetime` 转换为
 `datetime`

+ `resample('M').mean()`

+ `pd.date_range()` 生成一段时间的时间索引
  `pd.date_range('2005-01-05', '2018-05-05', freq='M')` 会自动对齐到月底

### `Series`
+ 可以用 `pd.Series({a:b, c:d})` 的方式创建

+ 对 `Series` 排序有
```py
s.sort_values() # 按值排序
s.sort_index()  # 按index排序
s.argsort()     # 排序的索引(int)
```

### `DataFrame`
+ 可以用 `pd.DataFrame({a:b, c:d})` 的方式创建

+ `drop` 方法可以选择性的丢弃某些列、行

+ `df.append(ignore_index=True)` 只能纵向合并，注意 `ignore_index` 参数

+ `pd.concat` 既能纵向也能横向合并两个数据框，注意 `ignore_index` 参数

+ `pivot()` 以及 `unstack()` 都可以将数据框进行行列转换。后者需要层级索引，可以通过 `MultiIndex.from_arrays()` `from_tuples` 等方法创建层级索引后再操作。而前者可以将指定列设置为 `index` 以及 `columns` 还有 `values`
```py
df.pivot(columns='Ticker', index='Date', values='UnitNV')
```

+ `df.to_csv(float_format='%.3f')`
  `date_format='%Y%m'` 用来设置时间

+ 保存为 `Excel` 文件
```py
writer = pd.ExcelWriter("result/持仓明细.xlsx")
full_info.to_excel(writer, "明细")
stat_info.to_excel(writer, "排名")
result.to_excel(writer, "报告")
writer.save()
```

**日期和日期偏移**
```py
from pandas.tseries.offsets import MonthEnd
from dateutil.parser import parse
print(parse('2018-01-07') - MonthEnd())
```
可以轻松获得 `2017-12-31`

**升采样与插值**
+ `reindex` 方法 + `interpolate` 方法
```py
new_sery = pd.concat([sery[points_low], sery[points_high]]).sort_index()
new_sery = new_sery.reindex(sery.index).interpolate('linear')
```

**随机采样**
+ `seris.sample(n=3)`

+ `Df.sample(n, frac, replace)`

+ 将 `grouby` 和 `sample` 组合到一起实现分组抽样
```py
def random_stocks(hs300, industry=None):
    if industry is None:
        return hs300.sample(n=100)
    return hs300.groupby(industry).apply(lambda x: x.sample(round(len(x)/3)))
```

## `scipy`
+ `savgol_filter` 用来做滤波

## `matplotlib`
**标注方法**
```py
plt.plot(sery)
plt.plot(curved_sery)
# plt.plot(sery)
plt.plot(sery[points_low], 'ks')
plt.plot(sery[points_high], 'gs')
plt.show()
```
