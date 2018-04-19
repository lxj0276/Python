## 实战经验

### 利用Python进行数据分析
**ipython**
+ ipython中 **读入路径还有中文的文件**， 增加参数 `engine='python'`

**pandas**
+ `data.columns` 可以快速获取列属性
+ `df.head()` 观察数据头

**`data.ix()` 已经被deprecated**
需要使用：
+ `df.loc[2]`
+ `df.loc[[2,3]]`
+ `df.iloc[2]`
+ `df.ix[2]` 当然这也仍然还是可以使用的
+ `irow, icol` 不存在了
+ `df.at[row,col]` **效率最高，取单个元素**
**也可以用来取出某一列或多列，语法是 `df.iloc[:,2]`**
**用df.loc取列最好用索引名，否则可能会报错**

**遍历行、列**
+ 使用`for index, row in df.iterrows():`
+ 使用`df.loc[i]`
+ 使用`df.itertuples()` 这个将每行放入tuples中
+ `df.iteritems()` 可以用于返回每一列

**pandas数据读入**
数据读入后会转换为pandas的数据类型，如`numpy.int64`，这在与MongoDB交互时，由于MongoDB只能写入数据，不能写入实例，将行转换为`dict`, 再对于数字 **18** 写入时，在python里是`numpy.int64`的实例，所以不能写入，但是直接转换为 `list` 则会变成int类型。

**pandas数据输出省略问题**
+ `pd.set_option('display.max_rows', len(x))`
+ 类似的可以设置打印输出的最大列数

**正则表达式**
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

**灵活利用Pandas和Numpy的向量化计算方式**
对于 **重复性的操作**，pandas和numpy提供了向量化的处理方式，比使用普通的python循环要快得多
```py
DAX['Ret_loop'] = 0.0 # 创建新列
for i in range(1, len(DAX)):
    DAX['Ret_loop'][i] = np.log(DAX['close'][i] / DAX['close'][i-1])
```
以上使用了numpy的向量化数值方法 `np.log`，但作用于普通的数值。实际上可以不用循环，进行更高效的运算
```py
DAX['Return'] = np.log(DAX['close'] / DAX['close'].shift(1))
```
这时传入 `np.log` 的参数是一个 **向量**，同时使用了 `series.shift(1)` 使得列移动一个索引位置

### 其他
**文件操作**
只读取数行，在针对大文件时，取小样本很有用
```py
with open('test.txt','w') as fw:
     for i in range(1000):
         fw.write(f.readline())
```
同时，读取 **大文件** 时，一次项读入十分消耗内存，可以利用文件迭代的特性
```py
for line in open('test.txt'):
    pass
```

**字典默认值**
对于普通的 `dict` 找不到键值会报错，有几种解决方案
+ `dict.setdefault(key, '')` 给key设定为默认值，如果key已经存在就不改变原来的值
+ `dict.get(key,'')` 在获取键key时，如果不存在key就设置新key，并且设置默认值
+ `collections.defaultdict(int)` 参数给定的是类型，将返回类型默认值
也可以传入一个 **无参函数规定默认值**

可以用来 **实现字典计数**
```py
l = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
d = defaultdict(int)
for i in l:
    d[i] += 1
sorted(d.keys(), key=lambda obj:d[obj], reverse=True)[0]
# 寻找value最大的键
```
