## 第5章
**高阶函数**
接受函数为参数，或是把函数作为结果返回的函数是高阶函数。
```py
def factorial(n):
    return 1 if n<2 else n*factorial(n-1)

fact=factorial # 可以把函数赋值给变量fact，体现了函数一等性

map(fact, range(11)) # 对列表中的所有元素执行映射操作

from operator import add
from functools import reduce
reduce(add,range(100)) # reduce返回执行某个操作替代sum

all(iterable) # 每个元素都是真值，返回True
any(iterable) #　只要有真值，返回True
```

**匿名函数**
```py
sorted(fruits,key=lambda word:word[::-1])
# lambda表达式
```

**可调用对象**

用户可以把一个对象声明成可以调用的对象，相对于c++中把类包装成函数
```py
class BingoCage:
    def __call__(self):
        return self.pick()
    # 定于了这个内置方法后，类拥有函数的行为
```

**定位参数和仅限关键字参数**

`*`可以对应多个无名参数，`**`可以对应多个形如`key=value`的参数对
```py
def tag(name,*content,cls=None,**attrs)
    return name
# name参数后面的参数计入到content数组里，而后面的键值对记录到attrs字典中

my_tag={'name':'img','title':'Sunset Boulevard',
        'src':'sunset.jpg','cls':'framed'}
tag(**my_tag)
# 用**展开字典时，里面键值对会自动对应成key=value传入函数中，即name=img等
# 因此在这个形式下content没有值
```

**函数式编程**

得益于`operator`和`functools`包的支持，能够进行函数式编程
```py
from functools import reduce
def fact(n):
    return reduce(lambda a,b:a*b, range(1,n+1))
# 单独用reduce，用到了匿名函数，不简洁

from operator import mul
def fact(n):
    return reduce(mul,range(1,n+1))
# 利用operator中的mul变得简洁一些

from operator import itemgetter
for city in sorted(metro_data,key=itemgetter(1)):
    print city
# itemgetter生成一个函数，itemgetter(1)实际上是取对象的第一个位置
# 相当于lambda fields:fields[1]

cc_name=itemgetter(1,0)
# 可以传多个参数，返回提取出的值构成元组，cc_name实际上是个函数，取1，0位置返回元组
# itemgetter使用[]运算符，支持任何使用__getitems__方法的类

from operator import attrgetter
name_lat=attrgetter('name','coord.lat')
# attrgetter与itemgetter类似，不过接受的参数是属性

from operator import methodcaller
upcase=methodcaller('upper')
# 与前两个类似，只是这个是取对象调用指定的方法,返回后还是一个函数
hiphenate=methodcaller('replace',' ','-')
# 多余的参数可以用于绑定参数，绑定部分参数，生成一个心函数

```

**functools.partial**
它可以基于现有函数创建一个新函数，新函数固定了部分参数，只接受一部分参数,和methodcaller类似
```py
from operator import mul
from functools import partial
triple=partial(mul,3)
# 绑定了需要两个参数的mul其中一个参数3，使它固定成翻三倍的函数

picture=partial(tag,'img',cls='pic-frame')
# 用于之前tag函数上，生成一个专门做picture标签的函数
```
