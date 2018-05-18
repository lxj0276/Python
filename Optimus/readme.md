# 编程手记
**函数类命名不要太详细太冗杂，关键信息相信就可以了**

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

**改动**
+ 删除 `hist_residuals()`
```py
    def hist_residuals(self, factor_returns):
        """
        get history residuals from regression results
        :param factor_returns: DataFrame, factor returns as the regression results
        :return: DataFrame with index as dates and columns as companies
        """
        # group by date
        date, returns, company, factor = list(self.names.values())[:4]
        grouped = self.x.groupby(self.x[date])

        # get residuals respectively
        def f(x, params):
            return x[returns] - (x[factor] * next(params)).sum(axis=1)

        results_residuals = None
        try:
            g = (list(factor_returns[factor].iloc[i]) for i in range(len(factor_returns)))
            results_residuals = pd.DataFrame([list(f(group, g)) for _, group in grouped])
        except StopIteration:
            pass

        results_residuals.columns = self.x[company].unique()
        return results_residuals.applymap(lambda x: 0.0 if x < 0.01 else x)
```
+ 不需要预测 `factor_loading`
```py
    def predict_factor_loadings(self, method, arg=None):
        """
        predict factor loadings
        :param method: method to predict
        :param arg: additional parameter used in prediction
        :return: DataFrame with index as company and columns as factors
        """
        company, factor = list(self.names.values())[2:4]
        if method == 'average':
            predicts = self.x[factor].groupby(self.x[company]).mean()
        elif method == 'ewma':
            def f(x, a):
                return x.ewm(alpha=a).mean()[-1:]

            predicts = self.x[factor].groupby(self.x[company]).apply(f, a=arg)
            predicts.index = self.x[company].unique()
        else:
            raise ValueError("predict_factor_loadings:undefined method" + method)
        return predicts
```
+ `risk_structure` 不需要考虑股票的协方差
+ 回归时可能会出现多重共线性

## numpy
+ `np.dot` 可以用于矩阵乘法，但是只限于两个矩阵相乘
+ `np.cov` 默认是行与行之间求协方差，要求列的协方差需要 **先转置**
+ `np.asmatrix` 后有了 `T` 属性用于转置
+ `np.append` 用于往 `array` 里面添加元素， 记得 **往回赋值**

## Pandas
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

**pandas cumulative function**
+ `pd.expanding_apply()`
+ `functools.reduce(lambda x,y:0.5*x+0.5*y, s)`

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
