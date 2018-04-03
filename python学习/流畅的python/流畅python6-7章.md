### 第六章

#### 经典的策略模式
策略模式为利用一个公共的虚基类，实现不同方法的统一接口，如在电商打折时，利用 `Promotion` 做接口，具体的细分方法只需要继承这个虚基类，重写 `discount()` 方法。

```py
from abc import ABC
# ABC 虚基类
class Promotion(ABC):
    # 新的虚基类继承ABC
    @abstractmethod
    def discount(self, order):
        """返回折扣金额"""
    # 利用了ABC中的装饰器，将方法装饰为虚方法
```

这个模式可以使用函数式编程的思想实现。
```py
class Order:
    def __init__(self,customer,cart,promotion=None):
        self.customer=customer
        self.cart=list(cart)
        self.promotion=promotion
    ...

def fidelity_promo(order):
    """函数以order类作为参数，可以直接作为参数初始化Order"""

promos=[f_promo, bulk_promo,large_promo]
def best_promo(order):
    return max(promo(order) for promo in promos)
# 找到最佳的折扣方案

promos=[globals()[name] for name in globals if
        name.endswith('_promo') and name != 'best_promo']
# 用globals()获取全局下所有的命名，取出按照规则命名的，加入promos中
# 有可能有些函数没有按照规则命名

import inspect
promos=[func for name,func in inspect.getmembers(promotions,inspect.isfunction)]
# 利用inspect选取函数，并且在单独promotions模块中选取
```

### 第七章

#### 装饰器基础知识
装饰器是可调用的对象，参数是另一个函数，处理这个函数，然后将它返回，包装成另一个函数。
**装饰器会在导入模块时立即执行。**
```py
@decorate
def target():
    print('running target()')

# 等同于
target=decorate(target)
```

#### 使用装饰器改进策略模式
```py
promos=[]

def promotion(func):
    promos.append(func)
    return func

@promotion
def fidelity(order):
    """"""

@promotion
def large(order):
    """"""
# 利用装饰器把函数注册到函数列表当中，只需遍历遍历该列表即可
```

#### 变量作用域规则
**即便是全局变量，如果是在函数内对它赋值，则认为是局部变量，需要先定义在使用。**
```py
b=6
def f(a):
    print(a)
    print(b)
    b=9
# 将会报错因为函数第三行对b赋值了，认为局部变量，需要先定义在使用。

def f1(a):
    global b
    print(a)
    print(b)
    b=9
# 如果想在赋值时作为全局变量，则需要事先声明global
```

#### 闭包
```py
def make_averager():

    # 从这里开始
    series=[]

    def averager(new_value):
        series.append(new_value)
        total=sum(series)
        return total/len(series)
    # 从这里结束，称为闭包
    # 闭包延伸到函数作用域之外，包含了series绑定
    # 调用make_averager()后本地作用域一去不复返
    # 但生成的函数会保存series

def make_averager():
    count = 0
    total = 0

    def averager(new_value):
        nonlocal count, total
        count += 1
        total += new_value
        return total / count

    return averager
# 必须使用nonlocal关键词，否则因为出现赋值，会被认为是局部变量
```

使用 `nonlocal` 关键词使得函数能够使用不在定义域内的变量，更好的利用闭包。

#### 实现一个简单的装饰器

```py
import time

def clock(func):
    def clocked(*args):
        # 接受任意定位参数，接收的参数和原来的func接收参数相同
        t0 = time.perf_counter()
        result = func(*args) # 得出结果
        elapsed = time.perf_counter() - t0
        name = func.__name__ # 得到函数名字
        ...
        return result
    return clocked

```

这个写法的缺点在于**不支持关键字参数**，因此可以利用 `functools.wraps` 装饰器把相关属性，并且能够处理关键字参数。

```py
import functools

def clock(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
    # 能够接收关键词参数
```

#### 标准库中的装饰器
`functools.lru_cache` 实现了备忘功能，是一项优化技术，把耗时的函数的结果保存起来，避免传入相同的参数时重复计算。`lru` 即 `Least Recent Used` ，缓存大小可以自己设置，最好是2的倍数。

```py
import  functools
from clockdeco import clock

@functools.lru_cache() # 装饰器工厂函数
@clock
def fibonacci(n):
    """求斐波那契数列"""
# 因为使用字典保存结果，所以函数参数必须是可散列的。
```

`functools.singledispatch` 可以用于重载方法或函数，将一个普通函数装饰为**泛型函数**，即接受不同类型的参数，并相应作出不同的操作。

```py
from functools import singledispatch
from collections import abc
import numbers
import html

@singledispatch
def htmlize(obj):
    """这是处理obj的基函数"""

@htmlize.register(str)
def _(text):
    """处理text"""

@htmlize.register(numbers.Integral)
def _(n):
    """处理整数"""

@htmlize.register(tuple)
@htmlize.register(abc.MutableSequence)
def _(seq):
    """序列"""

# 装饰之后的装饰器有了register方法，register接受的参数即为各种类型
# 装饰器装饰的函数不需要名字，可以用_()简化
# 装饰器可以叠放，使得同一函数接收不同参数
```

#### 参数化装饰器

```py
def register(active=True):
    def decorate(func):
        if active:
            """利用传入的参数"""
        else:
            """进行不同的处理"""
        return func
    return decorate

@register(active=false)
def f1():
    """函数一"""

@register()
def f2():
    """函数二"""

# 接收参数的装饰器实际上是个装饰器工厂函数，接收参数返回真正的装饰器
```
