# TriPeriods

**注释也要对齐哦！**

## python
+ `a[::-1]` 倒序

## numpy
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
