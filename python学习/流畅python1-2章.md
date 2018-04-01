### 第一章
python中有很多内置函数，内置函数可以用来做运算符重载、类初始化等等，使得python的类能够和标准的集合类型一样处理，即代码风格统一。如：
```py
__getitem__() # 重载【】运算符
__len__() # 即len()时会调用
__repr__() # 在控制台上时直接打印
__str__() # print时调用
__doc__() # 用于编辑说明文件
__contains__() # in运算符，默认按顺序做一次迭代搜索
__init__() # 初始化

# random里的choice包可以用于随机选择
from random import choice
choice(deck) # 在桌上随机抽一张牌出来
```

**注意在使用repr打印的时候可以规定打印的风格如：**
```py
def __repr__(self):
    return ' Vector(%r,%r)' % (self.x,self.y)
    # 其中后面括号中的元组代替%r的位置
```

**算数运算符的重载：**
```py
__add__() # 加法
__mul__() # 乘法
__abs__() # 绝对值
__sub__() # 减法
__bool__() # 布尔
```

### 第二章
可变序列<br>
 list、bytearray、array.array、collections.deque、memoryview

不可变序列<br>
 tuple、str、bytes

**列表推导**
```py
codes=[ord(symbol) for symbol in symbols] # 列表推导,ord变成unicode码

beyond=[ord(s) for s in symbols if ords(s) > 127]
# python支持三元运算符，可以在后面增加筛选条件

beyond=list(filter(lambda c:c>127, map(ord,symbols)))
# filter+map也可以取得相同效果
# filter返回布尔值，map表示某种映射

tshirts=[(color,size) for color in colors for size in sizes]
# 可以多重for
```

**生成器表达式**
```py
array.array('I', (ord(symbol) for symbol in symbols))
# 和列表推导类似，但节省雷村，逐个产生严肃，而不是先产生列表在传递到构造函数

for tshirts in ('%s %s' % (c,s) for c in colors for s in sizes):
    print(tshirt)
# %r和%s就是 repr()和str()的区别
```

**拆包**
```py
for country,_ in travel:
        print(country)
# _为占位符，当不需要使用的时候，可以这么用

a,b=b,a # 逗号运算符产生元组
# 这种写法很优雅，无需产生新对象就交换了两个变量的值，也可以它作为函数返回

t=(20,8)
divmod(*t)
# *可以把可迭代对象编程函数参数
a,b,*rest=range(5)
# *可以处理剩下的元素，剩下的数被自动归入rest中，变成一个list
# 元组拆包是可以嵌套的
```

**具名元组**
```py
Card=collections.namedtuple('Card',['rank','suit'])
City=namedtuple('City','name country population coordinates')
# 第一个参数为类名，第二个参数是所含属性的列表
City._fields
# 返回所有属性名称
```

**切片**
```py
s[start:end:step]
# 表示从start到end，每隔step切出来一个，其中三个参数都可以省略
# end不包含

x[i:...] # 是x[i,:,:,:]的缩写

l[2:5]=[20,30] # 必须用列表类型赋值，数量不一定相等
del l[5:7] # 删除

```

**序列的+和*运算**
```py
l*5 # 复制5遍列表中的元素，如果元素是引用需要特别注意

board=[['_']*3 ]*3 # 实际上创建了3个引用
# 修改其中一个，会导致3初同时修改
board=[['_']*3 for i in range(3)] # 这是正确的
```

**bisect和insort**
```py
bisect.bisect(list,value) # 默认相等值在右边
# 省略lo和hi参数，用于缩小搜寻范围

bisect.insort(list,item) # 用于在顺序序列中插入
```

**memoryview**
内存，用不同方法解释
