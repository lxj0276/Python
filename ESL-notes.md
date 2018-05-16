# ESL

## Overview of Supervised Learning
两种常见方法：
+ **线性模型** ，最小平方和
  假定了 `Y~X` 的线性关系
+ **最小紧邻(NN)**
  不需要假定关系，认为在近邻内，X是为常数的，因此近邻可以代表X
  + **维度灾难**
    1.立方体中要运用 `1/1000` 的数据空间，以及覆盖 `1/10` 的数据
    2.若 `N=100` 代表了一个稠密的 **1维** 输入空间，在 **10维** 时，需要 `100^10` 个数据点才能达到相同密度的输入空间，因此，高维时数据通常是 **稀疏的**
  + 在已知 **Y~X** 关系时不是最佳选择

三个概念：
+ `Statistical Models` 具有统计意义的模型比如，$Y=f(X)+\epsilon$ 利用残差项的分布假设
+ `Supervised Learning` 监督式学习是通过一个 `teacher` 的监督下利用 `Learning Algorithm` 不断根据 $y_i-f(x_i)$ 来修正自己
+ `Function Approximation` $f_\theta(x)=\sum\limits_{k=1}^{K}h_k(x)\theta_k$
  估计了一系列 $h_k(x)$ 的系数 $\theta_k$，

`Roughness Penalty`
+ 函数 `f` 不平滑，在某些区域变化非常快，则 `RSS` 会变成 `PRSS` 即对函数 `f` 施加惩罚，比如对二阶导数施加惩罚
  $PRSS(f;\lambda) = RSS(f)+\lambda \int[f^{''}(x)]^2dx$

`Kernal function`
+ $K_\lambda(x_0,x)$， 对 $x_0$ 附近的点 $x$ 赋予权重

`Bias-Variance Tradeoff`
随着模型复杂度提高，**误差变小，方差提高**
+ 对于 **训练集** 而言，模型复杂度提高，**预测偏差肯定会下降**
+ 对于 **测试集** 而言并非如此，预测偏差 **先下降后提高**，原因是 **过拟合**
找到 **测试集预测偏差最小** 的模型是非常关键的
