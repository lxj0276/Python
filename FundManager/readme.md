# readme

## python

+ `for`  **范围循环陷阱**
```py
for i in some_list:
  i = 0
```
赋值会失败，不会改变原来list中元素的值，要想改变原来list中的值，需要使用下标遍历的方式

## matplotlib

+ `plt.close()` 进行一次绘图的之后调用，以免重复绘图影响之后的绘图结果
