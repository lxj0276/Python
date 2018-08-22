# 编程手记
**函数类命名不要太详细太冗杂，关键信息相信就可以了**

## Python
+ `Warnings`
```py
import warnings
warnings.warn("")
```
+ `assert` 关键词，用来确信某一表达式是真的，在为假时报 `AssertionError`
```py
assert len(mylist) >= 1
```
+ `exec`
```py
exec 'print "Hello World"'
```

**装饰器装饰工厂函数**
+ 装饰器工厂函数接受某参数返回某装饰器
+ 装饰器接受函数，返回另一个函数，**原来的函数就不是本来的函数了**

## import
引入自己写的模块
```py
import sys;  
sys.path.append("路径名")  
import model
```

## Optimus
**注意日期排序**
**循环的效率总是很低，要考虑内部方法**
```py
def is_industry(col):
    if {i for i in 0, 1}.issuperset(set(col)):
        return True
    return False
```
+ `risk_structure` 不需要考虑股票的协方差
+ 回归时可能会出现多重共线性

## numpy
+ `np.dot` 可以用于矩阵乘法，但是只限于两个矩阵相乘
+ `np.cov` 默认是行与行之间求协方差，要求列的协方差需要 **先转置**
+ `np.asmatrix` 后有了 `T` 属性用于转置
+ `np.append` 用于往 `array` 里面添加元素， 记得 **往回赋值**

## Pandas
+ `astype` 可以转换数据类型，也可以在创建时指定

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

**合并两个DF**
+ `Df.append(ignore_index=True)` 只能纵向
+ `pd.concat` 既能纵向也能横向

**存储时设置**
+ `DataFrame.to_csv(float_format='%.3f')`
+ `date_format='%Y%m'` 用来设置时间

**画图**
```py
    import matplotlib.pyplot as plt
    s = pd.Series(r, index=sigma)
    s.plot()
    plt.show()
```

**options**
+ `pd.set_options()`

**索引**
+ `Index` 有 `append`,`delete`,`drop`,`insert` 以及一些集合方法
+ 利用pandas的算术功能时，**索引自动对齐** 会影响道计算结果，应该去除这个影响
+ 赋值时用 `Series` 对一个 `DataFrame`  赋值索引会对齐 **DataFrame** 的索引，即便长度一样也会产生NA。
+ `s1.index.union(s2)` 会使用 `s2` 的值和 `s1` 求并集
+ `reindex` 后要赋值回去，**否则不会改变**
+ `reindex` 和 `index=` 的区别是前者是重采样而后者是更换
+ `Index` 只有 `map` 方法没有 `apply` 方法
+ 只有 `Series` 有 `asfreq()` 方法
+ 只有 `Series` 以及 `DataFrame` 有 `resample` 方法
+ 多重索引有 `to_frame()` 方法方便转换为正常的 `DataFrame`
+ 多重索引有 `swaplevel([0, 1])` 方法可以转换 `level`
+ 多重索引有 `names` 属性
+ 可以直接多使用多重索引的数据框使用 `swaplevel`，这样可以用 `.loc[]` 选取想要的维度

**分组和聚合**
+ `pd.qcut` 参数： `labels=False` `10`
+ `apply` 运算多余的参数应该以 `kwargs` 的方式传入
+ `apply(print)` 会 **多print一次**
  需要用 `for` 循环解决
  ```py
    for guestid, df_group in grouped:
      print(list(df_group['guestid']))
  ```
+ 可以结合 **生成器**，每次给 `apply` 传入不同的参数
  ```py
  def f(x, params): print((x[factor] * next(params)).sum(axis=1))
    results_residuals = None
    try:
      g = (list(factor_returns[factor].iloc[i]) for i in range(len(factor_returns)))
      for _, group in grouped:
        f(group, g)
      except StopIteration:
        pass
  ```
+ `applymap` 元素级的运算
**时间序列索引**
+ `pd.to_datetime()`
+ `dateutil.parser.parse`
+ 可以用 `astype('str')` 来进行字符串的转换，再用 `pd.to_datetime` 转换为 `datetime`
+ `resample('M').mean()`
+ `pd.date_range('2005-01-05', '2018-05-05', freq='M')` 会自动对齐到月底

**pandas cumulative function**
+ `pd.expanding_apply()`
+ `functools.reduce(lambda x,y:0.5*x+0.5*y, s)`
+ `df.rolling.apply()`

**ewma**
+ `pd.ewma` 有 `alpha` 等参数
+ `x.ewm().mean()` 可以用 **lambda表达式** 传入 `apply` 方法

## Statsmodels
**OLS**
传入 `Y(Array)`, `X(Dataframe)`
```py
model = sm.OLS(Y,X)
results = model.fit()
results.params
```

## cvxopt
**安装**
+ 卸载 `pip uninstall numpy`
+ 安装 `cvxopt` `cvxopt-1.1.9-cp36-cp36m-win_amd64.whl`
+ 安装 `numpy + mkl` `numpy-1.13.3+mkl-cp36-cp36m-win_amd64.whl`
+ 可以直接使用 `pip install` 命令完成上述安装

**matrix**
+ `Q = 2 * matrix([[2, .5], [.5, 1]])`
  默认列表中每一个 `[]`， 都是矩阵中的一列
+ `A = matrix([1.0, 1.0], (1, 2))`
  第二个参数的元组则给出了矩阵的形式，这里是 **1*2 即一行两列**
+ 要解的 `x` 向量为 **列向量**， 方便添加约束
+ 将 `numpy` 的 `ndarray` 直接转换为 `matrix` 时，不会按照原生列表的方式解析，而是遵循 `numpy` 的方式。
+ `pandas` 的 `Dataframe` 不能直接转换为 `matrix`， 需要先经过 `np.asmatrix` 或 `np.asarray` 转化为 `numpy` 对象

**优化求解**
+ 二次规划问题 `solver.qp`
+ 目标为线性函数，约束为二次型 `solver.cpl` **注意要有解**
+ `cpl` 的 `F` 函数里面不要胡乱规定函数 **定义域**
+ `Ax=b`  **不能有重复的条件！** 会报错 `rank(A)<p`

**矩阵求导**
+ 一阶为 **列向量** （纵向扩展）
+ 二阶为 **矩阵** （对列上的每一个值，横向扩展）
