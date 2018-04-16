# 高效的Python

## 高效的数量上计算复杂的运算
**案例一**
Python本身提供了优秀的运算能力

```py
loops = 25000000
a = range(1, loops)
def f(X):
  return 3 * log(x) + cos(x) ** 2
%timeit [f(i) for i in a]
```
只需要 **15秒** 完成 **2500次** 运算
但使用 `Numpy` 能够更加优秀的完成任务，将时间缩短到 **1.7秒**
```py
import Numpy as np
a = np.arange(1, loops)
r = 3 * np.log(a) + np.cos(a) ** 2
```
已经很快了，但还能继续优化，有一个叫做 `numexpr` 专门用来改善 `numpy` 的性能
```py
import numexpr as ne
ne.set_num_threads(1)
f = '3 * log(a) + cos(a) ** 2'
r = ne.evaluate(f)
```
这样能够降低到   **1.2秒**，但还不够，`numexpr` 内建执行并行运算的功能，**使用全部线程** 进行运算，
```py
ne.set_num_threads(4) # 线程数为4
r = ne.evaluate(f)
```
这能下降到 **0.5秒** ！！

**案例二**
考虑很大的两个 `pandas` `DataFrame` 的复制，或运算。
+ 用 `append` 逐行添加，效率极慢。**事先决定dataframe的大小提供性能**
+ 逐行赋值，用 `df.loc[0]` `df.iloc[]` ，慢。用 `df.at[1,'time']` 逐个添加更快！
+ 因为 `at` 的效率是 `loc` 的 **1000倍**

### 总结
需要高效的运算有以下途径：
**案例一：**
+ Python本身提供的便利。
+ 某些库提供 **专业性的便利**，如 `numpy` 在数组运算上本身就有优化， **在金融序列数据的应用尤为常见**
+ 某些专门为优化性能而存在的库、库函数 `functools.lru_cache()` 函数和 `numexpr` 库， `Cython` 库
+ **并行、多线程**，使用多个线程并行运算

**案例二：**
+ 库提供的不同方法效率是有差别的，**根据需求，并了解提供的方法的效率是关键**
+ `df.at[]` 效率是 `df.loc[]` 的 **1000倍**

## Numba库
`Numba` 是开源、`Numpy` 感知的优化Python代码编译器。考虑一个两层循环 `5000*5000`，返回计数总循环次数的数量，涉及到 **25000000** 次计算。
**版本一**
```py
def f_py(I,J):
    res = 0
    for i in range(I):
        for i in range(J):
            res += int(cos(log(1)))
    return res
```
这是最慢的版本，明显用Numpy向量化处理可以更高的效率
**版本二**
```py
def f_np(I,J):
    a = np.ones((I,J), dtype=np.float64)
    return int(np.sum(np.cos(npp.log(a))),a)
```
用Numpy可以获得更高的效率，**但也消耗更多的内存**
**版本三**
```py
import numba as nb
f_nb = nb.jit(f_py)
%timeit f_nb(I,J)
```
由于产生了可调用编译版本，**可以获得更高的性能**
并且 `Numba` 支持 `Numpy` ，即便函数中已经用 **`Numpy` 向量化** 的方法进行了优化，也还是可以利用 `Numba` 再度提高效率。

### 总结
+ **效率**，一行代码，原始函数不需要改变
+ **加速**，无论是 `numpy`向量还是纯python，都可以提高速度
+ **内存**，不需要初始化大型数组对象，**编译器专门为问题生成了机器代码**，维持和纯python相同的内存效率

## Cython
**有待了解**

## 用GPU进行蒙特卡罗模拟
**尽请期待**
