## 数据分组和聚合运算
`拆分-应用-合并` 的思考和运作方式。

### 分组
**groupby技术**
+ `df.groupby('key')`
+ `df.groupby(df['key'])`
+ `df.groupby(['key1', 'key2'])`

`goupby` 对象支持迭代，由分组名和数据块构成
```py
for name, group in df.groupby('key1'):
    print(name)
    print(group)

for (k1, k2), group in df.groupby(['key1', 'key2']):
    print(k1, k2)
    print(group)

pieces = dict(list(df.groupby('key1')))
# 转换为词典有利于之后的操作
```

**选取特定的一个或一组列**
+ `df.group('key1')['data1']` 一列
+ `df.group('key1')[['data1', 'data2']]` 一组列
+ 是**语法糖**

**用字典和Series等映射也可以完成分组**

**通过函数分组**
+ `people.groupby(len).sum()`
+ 返回值作为分组的名称

**根据索引级别分组**
+ `data.groupby(level='city', aixs=1).count()`
+ 传入level关键词参数，可以是索引名和层级编号

### 聚合
**面向列使用多个自定义函数聚合**
+ `grouped.agg(f, 'mean')` 用 `'mean'` 表示求均值，传入的 `f` 是自定义的
+ 可以自己给定列名，通过传入一个元组列表 `[('foo', 'mean'),('bar', np.std)]`
+ 这时前面为**列名**，后面为**具体的函数**

**`trasform` 方法**
这个方法把一个函数应用到各个分组，并**返回到原来适当的位置中**去
+ `grouped.groupby(key).mean()`
+ 输出结果为：
    ```py
    group1 1 1 1 1
    group2 2 2 2 2
    ```
+ `grouped.groupby(key).transform(np.mean)`
+ 输出结果为：
    ```py
    group1 1 1 1 1
    group2 2 2 2 2
    group1 1 1 1 1
    group2 2 2 2 2
    ```

** `apply` 方法**
这是 **一般性** 的分组合并方法
```py
# 例1，用各组平均值填充各组的缺失值
fill_mean = lambda g:g.fillna(g.mean())
data.groupby(key).apply(fill_mean) 
