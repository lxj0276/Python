# 编程手记
**函数类命名不要太详细太冗杂，关键信息相信就可以了**

## Pandas
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

**时间序列索引**
+ `pd.to_datetime()`
+ `dateutil.parser.parse`

**pandas cumulative function**
+ `pd.expanding_apply()`
+ `functools.reduce(lambda x,y:0.5*x+0.5*y, s)`

## Statsmodels
**OLS**
传入 `Y(Array)`, `X(Dataframe)`
```py
model = sm.OLS(Y,X)
results = model.fit()
results.params
```
