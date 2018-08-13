# readme

## python

+ `for`  **范围循环陷阱**
```py
for i in some_list:
  i = 0
```
赋值会失败，不会改变原来list中元素的值，要想改变原来list中的值，需要使用下标遍历的方式

+ 在命令行执行 `python` 脚本时
>`python Run.py` 与 `python /home/path/Run.py` 执行结果是不一样的
因为在执行的过程中，命令行默认只是搜索当前执行路径，所以会导致一部分引用的模块找不到

因此要使用 `os` 和 `sys` 模块
`sys.argv[0]` 获取当前执行文件的目录
`os.abspath` 取得绝对路径
需要在脚本前的相对路径加上绝对路径

+ `datetime` 模块获取时间
`date.today()` 今日
`datetime.now()` 当前 

## matplotlib

+ `plt.close()` 进行一次绘图的之后调用，以免重复绘图影响之后的绘图结果
