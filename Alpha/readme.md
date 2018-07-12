# readme

安装国内的源
```
pip install web.py -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
```
## ubuntu + jupyer notebook
```py
# 创建密码
python -c "import IPython;print IPython.lib.passwd()"

jupyter notebook --generate-config --allow-root
vim ~/.jupyter/jupyter_notebook_config.py

# 修改配置文件
c.NotebookApp.ip = '*'
c.NotebookApp.allow_root = True
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
c.NotebookApp.password = u'刚才生成的密文(sha:...)'
c.ContentsManager.root_dir = '/data/jupyter/root'

# 后台启动
nohup jupyter notebook > /data/jupyter/jupyter.log 2>&1 &
```
详见 https://www.cnblogs.com/douzujun/p/8453030.html

### notebook
+ `Esc` 进入 `command mode`
  `Enter` 进入 `Edit mode`
+ `command mode` 下
  `Shift + Enter` 运行当前 `shell` 移动到下一个 `shell`
  `a` 在之前创建一个 `cell`
  `b` 在之后创建一个 `cell`
  `y` 代码模式
  `m` markdown模式
  `1` 显示行数
  `s` 保存
  `d` 删除
  `ctrl + Shift + p` 启动命令面板
  `h` 快捷键帮助 

## python
+ `*` 符号的使用
```py
def make_group(num):
    index = [list(combinations(range(num), i)) for i in range(1, num+1)]
    return list(chain(*index))
```

## numpy
+ `matrix` 的索引方式和 `ndarray` 不同
```py
matrix[0, 0]
ndarray[0][0]
```

## pandas
+ 对 `Series` 排序有
```py
s.sort_values() # 按值排序
s.sort_index()  # 按index排序
s.argsort()     # 排序的索引(int)
```

+ `pivot()` 以及 `unstack()` 都可以将数据框进行行列转换。后者需要层级索引，可以通过 `MultiIndex.from_arrays()` `from_tuples` 等方法创建层级索引后再操作。而前者可以将指定列设置为 `index` 以及 `columns` 还有 `values`
```py
df.pivot(columns='Ticker', index='Date', values='UnitNV')
```

+ `dropna(inplace=True)` 对目前的数据框进行修改，不产生新数据框

## sql
+ 外大内小用 `exist`，外小内大用 `in`

+ `or` 的效率和 `in` 差不多，都很慢，数据量小的时候，不如直接多次 `=` 查询
