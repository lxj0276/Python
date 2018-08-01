# readme

## Python
+ `list.extend()` 后面接 `iterable` 用于多个值的扩展


## pandas
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

## statsmodels
+ `OLS()`
  `missing='drop'` 忽略缺失值

## matplotlib
+ `savefig()`
