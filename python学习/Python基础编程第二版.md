## Python基础编程的新知识

### 字符串方法
```py
str.replace() # 替代
str.strip() # 默认去两侧的空格，也可以指定要去掉的字符
str.translate(table) # 和makertrans连用，第二个参数指定要删去的字符
table = make_trans('as','cd') # 前者对应后者
```
### 异常之禅
+ `raise` 后面不接参数则继续向上抛出
+ `except` 后面不接参数就捕捉所有异常
+ 支持多个`except` 子句，也可以在一个子句中接受多个异常
+ 用 `try/except` 有时候可以比 `if/else` 更简洁和易读

### `re` 模块
+ 从输入的字符串到正则表达式要经过两层转义
    + 如 `'a/n'` 中就有转义字符,因此 `'\\'` 在正则中才是 `'\'`
    + 也可以用 `r'\.*'` 的方式表示原始字符串
+ 're.compile()' 用来给正则表达式加入更多的参数

### 文件处理
```py
import fileinput

for line in fileinput.input(filename):
    process(line)
# 懒惰行迭代，避免一次性读入内存太大

for line in open(filename):
    process(line)
```
### 有用的包
+ 图形界面：`TKinter` `wxpython` `PyQt`
+ 数据库：`SQLite` `redis` `cassandra` `MongoDB`
+ Python Web应用：`urllib2` `BeautifulSoup`
+ 性能及单元测试：`PyLint` `PyChecker` `unittest` `doctest`
+ 扩展python：`Jython` `CPython`
+ Excel: `xlsutils` `xlrd` `xlwt`

### 测试
**先测试，后代码**
+ 精确的**需求说明**，明确程序的目标，描述程序必须满足的需求
+ 为函数编写准备测试样例

**测试工具**：
`doctest` 用来检查函数，只要在它的文档字符串里面添加在交互解释器中的测试案例就可以了。
```py
def square(x):
    '''
    squares

    >>> square(2)
    4
    >>> square(3)
    9
    '''
    return x*x
# 这一段文档说明里面包含了测试样例

if __name__ == '__main__':
    import doctest
    doctest.testmod(my_math) # 假设模块名叫做my_math

$ python my_math.py -v # -v参数表示详述
```

`unittest` 可以做更加成熟的单元测试
