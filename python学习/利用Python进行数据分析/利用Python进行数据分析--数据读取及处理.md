## 文件读入
```py
df = pd.read_csv('a.csv', names=[], index_col='message')
# 给出读入的数据的列名和索引
# skiprows=[] 跳过某些行
# 那na_values=[]接受表示缺失值的字符串
# header表示用作列名的行号，默认为0

```
可以读入 `json` 格式的数据，注意读入的方式。

## 数据规整化
**dataframe合并**
```py
pd.merge(df1, df2)
# 默认是去重叠列名
pd.merge(df1, df2, on='key')
# 可以根据一个或者多个key合并
pd.merge(df1, df2, left_on='k1', right_on='k2')
# 可以分别给定左右的键
pf.merge(df1, df2, how='outer')
# 默认是inner，取交集，outer是并集，还可以用left，right
pf.merge(df1, df2, suffixes=('_l', '_r'))
# 可以指定未合并的相同列的后缀名
```
**索引上的合并**
+ `pd.merge`
+ `df.join`
```py
pd.merge(df1, df2, left_on='k1', right_index=True)
# 用索引，如果是层次式索引，则必须指定相应的列
df1.join(right2, on='key', how='outer')
# 这时默认左边是用索引
```

**轴向连接**
+ `pd.concat`
+ `df1.combine_first`
```py
# s1，s2，s3都是series
pd.concat([s1, s2, s3], aixs=1)
pd.concat([s1, s2, s3], aixs=1, join='inner')
# 这种方式将去行索引的并集
# 可以通过join_axes指定行索引
pd.concat([s1, s2, s3], keys=[])
# 默认列向连接，指定keys相当于增加层次索引
# 如果axis=1，相当于给定列名
# 同样的逻辑对于dataframe也是一样的，会增加层及索引

df1.combine_first(df2)
pd.where(pd.isnull(df1), df2, df1)
# 这两个作用相同
```

**重塑和轴向旋转**
+ `stack` 将列旋转为行
+ `unstack` 行旋转为列，并且默认操作的是**最内层**
+ `pivot` 用某一列数据作为索引，分别行索引，列索引，最后一个参数说明数列
```py
# 行索引是不会变的，列索引会变为行索引的最内层
# 如果传入分层级别的编号或者名称
result.unstack(0) # 最外层变为列索引
result.unstack('state') # 等价操作
# 可能引入缺失数据，但可逆
# 可以用 dropna=False 留下na

ldata.pivot('date', 'item', 'value')
# item是列名，一列是重复数据，可以作为索引
# 如果有两个数据列，则会生成层级索引
```

**数据转换**
+ `data.duplicate` 返回布尔型dataframe
+ `data.drop_duplicate` 丢掉重复行
+ `series.map()`
+ `data.replace()`
```py
data.drop_duplicate(['k1']) # 只看某个列的重复
data.replace([],[]) # 可以传入两个列表同时转换
data.replace({}) # 可以传入字典
```

**重命名索引**
+ `data.index.map` 要赋值回去完成修改
+ `data.rename(index=str.title, columns=str.upper)`

**离散化和组划分**
+ `pd.cut(ages, bins)`
```py
cats = pd.cut(ages, bins) # bins是一个列表，给定间隔，左开右闭
cats = pd.cut(ages, bins，labels=group_names) # 可以指定组名
cats = pd.cut(ages, 4， precision=2) # 均匀分成四组
pd.value_counts(cat) # 分组计数
```

**字符串方法**
+ `data.str.contains` 利用str属性访问pandas的字符串方法，并且是矢量化的字符串操作
