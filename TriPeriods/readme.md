# TriPeriods

**注释也要对齐哦！**

## pyinstaller
指定 `python` 解释器的位置，用于直接运行
```py
#! C:\Python\python.exe
```
在首行指定编码，允许中文路径
```py
# -*- coding:utf-8 -*-
```

## python
+ `a[::-1]` 倒序

**格式化输出**
```py
print('{} is {:.2f}'.format(1.123,1.123)) # 取2位小数
print('{0} is {0:>10.2f}'.format(1.123))  # 取2位小数，右对齐，取10位
```

## pandas
+ 存储字符串
```py
dtype = 'str'
.astype('str')
```

+ 保存为 `Excel` 文件
```py
writer = pd.ExcelWriter("result/持仓明细.xlsx")
full_info.to_excel(writer, "明细")
stat_info.to_excel(writer, "排名")
result.to_excel(writer, "报告")
writer.save()
```

+ 日期与日期偏移
```py
from pandas.tseries.offsets import MonthEnd
from dateutil.parser import parse


print(parse('2018-01-07') - MonthEnd())
```
可以轻松获得 `2017-12-31`

## numpy
**numpy不支持字符串存储**

**存储**
```py
np.savetxt('mydata/daily_dates.csv', daily_dates, delimiter=',')
```
可以指定 `fmt` 参数来指定存储的格式

+ `np.fft` 傅里叶变换
`np.fft.fft`:
`np.fft.ifft`: 傅里叶逆变换
`ndarray.conjugate()`: 求共轭
`ndarray.real`: 取实数部分

+ `np.pad()` 数组补
`pad_width = (1, 1)`: 两边
`mode = 'constant'` 常数

+ `np.exp()` 指数运算

+ `np.column_stack((a, b))`
`np.row_stack((a, b))` 合并
`np.insert()`

+ `np.linalg.lstsq(x, y)[0]` 线性回归取系数

+ `np.sort` 排序
`np.sort(a)[::-1]` 倒序
`np.argsort()` 索引排序

+ `np.nansum()` 含缺失值的求和
`np.nancumsum()` 一系列含有 `nan` 求值ֵ

+ `np.reshape(-1)` 多维转一维，同样也可以一维转多维 **不产生复制**

+ `matlab` 中 `std` 为 `ny.std(ddof=1)` 要设置自由度

## tkinter

+ 创建窗口
```py
import tkinter as tk

window = tk.Tk()
window.title('my window')
window.geometry('200x100')

# 这里是窗口的内容

window.mainloop()
```

+ `Label`
```py
l = tk.Label(window,
    text='OMG! this is TK!',    # 标签的文字
    bg='green',     # 背景颜色
    font=('Arial', 12),     # 字体和字体大小
    width=15, height=2  # 标签长宽
    )
l.pack()    # 固定窗口位置
```

+ 文字变量存储器
```py
var = tk.StringVar()    # 这时文字变量储存器

l = tk.Label(window,
    textvariable=var,   # 使用 textvariable 替换 text, 因为这个可以变化
    bg='green', font=('Arial', 12), width=15, height=2)
l.pack()

var.set('you hit me')   # 设置标签的文字为 'you hit me'
```

+ `Entry` `Text`
`Text` 有 `insert` 方法
```py
e = tk.Entry(window,show='*')
e.pack()

t = tk.Text(window,height=2)
t.pack()
```

+ `messagebox`
```py
def hit_me():
   tk.messagebox.showinfo(title='Hi', message='hahahaha')
```
