### 可迭代的对象、迭代器和生成器

#### 典型的迭代器
典型的迭代器实现了 `__next__()` 和 `__iter__()` 方法。
```py
# 这是在sentenceIterater的定义体里
def __next__(self):
    try:
        word = self.words[self.index]
    except IndexError:
        raise StopIteration()
    self.index += 1
    return word

def __iter__(self):
    return self
```

#### 迭代器第三版
在这个版本中，可以不单独在前一个版本中实现迭代器，并写 `__next__()` 方法，而是通过生成器函数，或者是生成器表达式完成。如：
```py
# 注意这是在sentence的定义体里
def __iter__():
    for word in self.words:
        yield word
    return
    # 这样定义__iter__()方法，就可以了
```
**这样就不需要单独顶提供一个迭代器类了！**

#### 生成器工厂
上面的生成器函数返回一个生成器对象，生成器函数就会变成一个生成器工厂。
```py
def f():
    yield 1
    yield 2
    yield 3

it = f() # it即为生成器
next(it) # 1
for i in f():
    # 可以放到for循环里
    pass
```

#### 惰性实现
惰性与急切相对，即只在需要的时候返回值，比如生成器，调用next方法每次返回一个值。在之前的sentence的实现里，一次性先把文本全部处理，再传入迭代器函数中，这不符合惰性原则。
`re.finditer` 是 `re.findall` 的惰性版本，返回的不是列表，而是一个生成器，按照需求生成 `re.matchObject` 因此可以节省大量内存，我们可以让类变得懒惰，仅在需要时生成下一个单词。
```py
# 这是在sentence的定义体中
def __init__(self, text):
    self.text = text

def __iter__(self):
    for match in RE_WORD.finditer(self.text):
        yield match.group()
```

#### 用生成器函数产出等差数列
```py
def aritprog_gen(begin, step, end=None):
    result = type(begin + step)(begin)
    forever = end is None
    index = 0
    while forever or result < end:
        yield result
        index += 1
        result = begin +step * index
```
除了用以上生成器函数产出生成器外，还可以使用 `itertools` 模块，这个模块提供了19个生成器函数，结合起来使用能实现很多有趣的用法。比如 `itertools.count` 能生成无穷多个数，我们提供可选的 `start` 和 `step` 值。不过 `itertools.takewhile` 函数则不同，会生成一个使用另一个生成器的生成器，在制定条件计算结果为false时停止 。
`gen = itertools.takewhile(lambda n: n<3, itertools.count(1, .5))`

#### 标准库提供的生成器函数
```py
# 用于过滤
itertools.compress(it, select_it) # 并行处理两个可迭代对象，后者中元素为真，则产出前者中对应的元素
itertools.dropwhile(predicate, it) # 按照predicate丢掉满足要求的元素
filter(predicate, it) # 留下满足要求的元素
itertools.takewhile(predicate, it) # 返回真值则产出，否则停止

# 用于映射
itertools.accumulate(it, [func]) # 类似于reduce但是在生成器水平上的，并且返回生成器
enumerate(iterable, start=0) # 产生有两个元素组成的元组，结构是index,item，index从start开始计数
map(func, [it1,it2...]) # 后面的it都是func的参数
itertools.starmap(func, it) # 后者的每个元素以 *it的形式传入func

# 用于合并
itertools.chain(it1,it2) # 无缝链接
itertools.product(it1,it2) # 笛卡儿积
itertools.zip(it1,it2) # 并行从输入的可迭代对象中获取元素，直到最短的为止

# 将输入的各个元素扩展成多个输出元素的生成器
itertools.combinations(it, out_len)
itertools.count(start=0, step=1)
itertools.cycle(it) # 按顺序重复不断
itertools.repeat(item) # 一直永续的产出同一个元素，除非提供数量

# 分组
itertools.groupby(it, key=None) # 默认按值相等分组
```

#### iter函数不为人知的用法
```py
def  d6():
    return randint(1,6)

d6_iter = iter(d6, 1)
# 持续调用d6，知道到标记值1为止
```

### 第十五章 上下文管理器和else块

#### 无处不在的else语句块
除了有经典的 `if/else` ，其实还有 `for/else` `while/else` `try/else`，他们的行为如下：
*  for，仅当循环运行完毕才执行
*  while，仅当因为条件为假而退出时，才执行
*  try，仅当没有异常抛出时才执行

#### 上下文管理器和with
上下文管理器对象的存在目的是管理with语句，上下文管理器协议包含 `__enter__` 和 `__exit__` 两个方法。with语句开始调用时在上下文管理器对象上调用 `__enter__` 方法，运行结束时调用 `__exit__` 方法。
```py
with open('mirror.py') as fp:
    src = fp.read(60)
```
`fp` 实际上是个上下文管理器对象。
