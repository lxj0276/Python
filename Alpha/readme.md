# readme

## python
+ `*` 符号的使用
```py
def make_group(num):
    index = [list(combinations(range(num), i)) for i in range(1, num+1)]
    return list(chain(*index))
```
