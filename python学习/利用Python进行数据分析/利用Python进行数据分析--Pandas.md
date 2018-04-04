## Pandas

### 基本数据结构
最基本的两个数据结构是 `Series` 和 `Dataframe` ，和数组相似，特别指出是带有索引，并且能够根据索引访问。
```py
obj = Series([4,7,5,-3], index=['d', 'a', 'b', 'c'])
obj2 = obj[['a', 'b']]
# 能够选取访问多个索引
obj2[obj2 > 0]
# Numpy的布尔索引，数组元素都保留了，同时还可以看成订场的字典

obj4 = Series(sdata, index=states)
# 可以传入新索引，缺失的值为Na
obj4.isnull()
obj4,notnull()
# 检测缺失值，并且返回布尔数组
```

**dataframe**
```py
frame = DataFrame(data, columns=[], index=[])
# 用columns指定列序列的顺序
# index表示行的索引
# 缺失值都会用NA表示

frame['state']
frame.year
# 两种方位列属性的方式

frame.ix['three']
# 访问行索引的方式
frame['debt'] = 16.5
# 进行赋值时仍然有传播性质
del frame['debt']
# 用del删除
frame.values
# 返回ndarray表示的数据
```

### 重要方法和基本功能
**重索引**
```py
obj,reindex([]) # 重新创建索引，并根据新索引重排
obj.reindex([], fill_value=0)
# 对于缺失值可以指定填充方式
obj.reindex([], method='ffill')
# 可以指定插值方法
obj.reindex(index=[], columns=[])
# 可以重新索引行或者列，插值只能应用于行
obj.ix([], states)
#　利用.ix实现更加简单
```
**丢弃**
```py
data.drop('a') # 默认是列
data.drop(['a','b']) # 一组列
data.drop(['two','four'], axis=1) # 通过axis参数可以丢弃行
```

**索引选取过滤**
```py
# 对于Series，可以按照整数选取，也可以按照索引名切片
s[[1,3]] # 选取一组
s[1:3]  # 切片
s['a':'b'] # 注意边界包含
s['a':'b']  = 1 # 传播

# 而对于DataFrame，则是索引列
data['two'] # 列或一组列

data[:2] # 如果是切片则是选择行了
data[data['three'] > 5] # 布尔型数组选取行
data[data < 0] = 5 # 布尔型dataframe索引
# 总结起来索引形式有 obj[val], obj.ix[val], 注意不同表现形式
# 也有icol。irow方法，按照位置选取
```

**算术运算和数据对齐**
```py
df1 +　df2
# 注意加法的时候会对齐
# 会引入NA

# 通过使用算术方法，可以设置天充值
df1.add(df2, fill_value=0)
df1.reindex(columns, fill_value) # 重新索引也可以指定填充

# dataframe和series的运算
arr - arr[0] # 在numpy里面是传播的
frame - frame.ix[0] # 在pandas也是
series = frame['d']
frame.sub(series, axies=0)
# 如果要减去某一列并传播，则必须这样写，axis指定轴
```

**函数应用和映射**
```py
f # 求最大最小的差
frame.apply(f) # 在列上应用，一列一列
frame.apply(f, axis=1) # 在行上应用，一行一行
frame.applymap(f) # 可以在元素级上
frame['a'].map(f) # 在series上是map方法
```

**排序**
```py
frame.sort_index() # 列方向的索引，即行索引排序
frame.sort_index(aixs=1, ascending=False) # 行方向上的索引，即列索引排序，默认升序

series.order() # 按值排series
frame.sort_index(by='b') # 按照某一列的值排序
# 传入索引列表，则会有多个标准

```
**汇总描述性统计**
```py
df.sum(axis=1, skipna=False) # 按照不同方向，是否跳过缺失值
#　还有level参数规定，层级
```

**相关系数协方差**
```py
returns.corr()
returns.cov()
#　自相关矩阵，和协方差矩阵

returns.corrwith(returns.IBM)
# 逐列求相关系数
```
