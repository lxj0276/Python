# readme

## python 与 R 交互

+ `rpy2` 的安装
```
https://www.lfd.uci.edu/~gohlke/pythonlibs/#rpy2
```
安装 `wheel` 文件

+ 使用
```py
import numpy as np
import os
os.environ['R_USER'] = 'C:/安装程序/R/bin'
os.environ['R_HOME'] = 'C:/安装程序/R'

def test():
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr
...
```
由于要求有环境变量，有两个选择：
+ 在 `IDE` 中加入环境变量，`Pycharm` 中为 `Run Configuration`
+ 通过 `os` 包加入环境变量，即 `os.environ['R_USER']`

## pandas
+ 快速获得每月初、每年初
```py
df.index = pd.to_datetime(df.index)
year = df.index.year
year_starts = year.to_series().diff()
year_starts = df.index[year_starts.fillna(1.0) > 0]
```
`index.to_series()` 转换为 `Series`
