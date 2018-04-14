## 实战经验
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
