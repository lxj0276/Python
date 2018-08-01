# readme

## Python
+ `list.extend()` 后面接 `iterable` 用于多个值的扩展


## pandas
+ `pd.Series`
  `copy=False` 默认创建时，是不复制的，对新对象的修改会影响到原来的对象

+ `pd.read_csv()`
  `header=[0, 1]` 用于指定多重索引
  `index_col=0` 用于指定索引所在列

+ `isnull().any()` 检测是否存在缺失值
  `notnull`

+ 保存至 `Excel`
```
writer = pd.ExcelWriter('output/风险平价.xlsx')
w0.to_excel(writer, str(risk))
w1.to_excel(writer, ...)
writer.save()
```

+ `df.divide(series, axis=0)`
  `axis=0` 这个参数可以使得df能够除以一列数，否则不行，直接使用 `/` 会得到错误的结果

+ `df.iterrows()`
  `df.itertuples()` 用来遍历行或者列

## statsmodels
+ `OLS()`
  `missing='drop'` 忽略缺失值

## matplotlib
+ `savefig()`
