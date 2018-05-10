# 编程手记
**函数类命名不要太详细太冗杂，关键信息相信就可以了**

## Pandas
**索引**
+ `Index` 有 `append`,`delete`,`drop`,`insert` 以及一些集合方法

**分组**
+ `pd.qcut` 参数： `labels=False` `10`

## Statsmodels
**OLS**
传入 `Y(Array)`, `X(Dataframe)`
```py
model = sm.OLS(Y,X)
results = model.fit()
results.params
```
