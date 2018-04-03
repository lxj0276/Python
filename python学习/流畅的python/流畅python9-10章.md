### 第9章

#### classmethod
`classmethod` 装饰器定义的是类方法，和实例方法不同，实例方法作用于类的实例，而类方法作用于类型本身。
```py
class Demo:
    @classmethod
    def klassmeth(*args):
        return args

Demo.klassmeth()
# 是在Demo类型上使用的方法
```

#### 可散列的类型
要是一个类的实例可以被散列，首先要求类是不能被修改的，即不可变。因此我们需要 `@property` 装饰器，把读值方法标记为特性，不能对其赋值。
```py
class Vector2d:
    typecode = 'd' # 这是一个类属性

    def __init__(self, x, y):
        self._x = float(x)
        self._y = float(y)

    @property # 把读值方法标记为特性读值方法与公开属性同名
    def x(self):
        return self._x
    ...
```
上面的 `typecode` 是类属性，要修改类属性，必须在类上修改，不能在实例上修改。
```py
Vector2d.typecode = 'f'

class ShortVector2d(Vector2d):
    typecode = 'f'
# 经常用于继承时定制类的数据属性
```

### 第10章
如果是多维的向量，我们需要同时取 `xyzt` 四个方向的分量时，按照前一章的写法，需要写四次 `@property` 非常麻烦，因此我们可以使用 `__getattr__` 特殊方法，属性查找失败后，会调用这个方法。
```py
# 还在类的定义体中
shortcut_names='xyzt'

def __getattr__(Self,name):
    cls= type(self) # 获取类型

    if(len(name)) == 1:
        pos = cls.shortcut_names.find(name)
        if 0 <= pos < len(self._components):
            return self._components[pos]

```
光这样写是不够的，错误会出现在给 `v.x` 赋值的时候，因为对象本身没有 'x' 这个属性，因此赋值时，会添加这个属性，并且以后每次进行访问的时候，都会直接访问新创造的属性。因此我们要利用 `__setattr__()` 特殊方法。
```py
def __setattr__(self,name,value):
    if len(name) == 1:
    """如果设置的属性为我们要存取的v.x之类的，方法不同"""
    ...

    super().__setattr__(name,value)
    # 如果不是，则使用默认行为，即在超类上调用
    # 非常重要
```
