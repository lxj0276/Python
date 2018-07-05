# readme

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
