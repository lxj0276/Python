from functools import reduce
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
# 正确安装 cvxopt 的方式见这里:https://blog.csdn.net/qq_32106517/article/details/78746517

show_progress = False


def ewma(x, weight):
    """
    :param x: Series
    :param weight: 权重
    :return: ewma结果
    """
    return reduce(lambda y, z: (1 - weight) * y + weight * z, x)


def max_returns(returns, risk_structure, risk, base=None, up=None, eq=None, noeq=None):
    """
    给定风险约束，最大化收益的最优投资组合
    :param returns: 下一期股票收益
    :param risk_structure: 风险结构
    :param risk: Double, 风险或是年化跟踪误差的上限
    :param base: Vector, 基准组合
    :param up: Double, 个股权重的上限
    :param eq: list, [A, B], 其余等式约束条件
    :param noeq: list, [G, h], 其余不等式约束条件
    :return: 最优化的投资组合
    """
    assert len(risk_structure) == len(returns), "numbers of companies in risk structure " \
                                                "and returns vector are not the same"

    if base is not None:
        assert len(base) == len(returns), "numbers of companies in  base vector and returns vector are not the same"

    if up is None:
        up = np.ones(len(returns))  # 默认取值为1
    else:
        up = np.ones(len(returns)) * up  # up 既可以为值也可以为列向量

    r, v = matrix(np.asarray(returns)) * -1, matrix(np.asarray(risk_structure))
    num = len(returns)
    base = matrix(np.asarray(base if base is not None else np.zeros(num))) * 1.0  # base 默认为0

    def func(x=None, z=None):
        if x is None:
            return 1, matrix(0.0, (len(r), 1))
        f = x.T * v * x - risk ** 2 / 12
        df = x.T * (v + v.T)
        if z is None:
            return f, df
        return f, df, z[0, 0] * (v + v.T)

    # 不能卖空
    g1 = matrix(np.diag(np.ones(num) * -1))
    h1 = base
    # 个股上限
    g2 = matrix(np.diag(np.ones(num)))
    h2 = matrix(up) - base
    g, h = matrix([g1, g2]), matrix([h1, h2])
    # 控制权重和
    # 0.0 if sum(base) > 0 else 1.0 在没有基准（即基准为 0.0）时为1.0，有基准时为0.0
    a = matrix(np.ones(num)).T
    b = matrix(0.0 if sum(base) > 0 else 1.0, (1, 1))

    if noeq is not None:
        assert type(noeq) == list and len(noeq) == 2, "不等式约束条件为形如 [G, H] 的列表"
        g3, h3 = matrix(noeq[0]), matrix(noeq[1])
        g, h = matrix([g, g3]), matrix([h, h3])

    if eq is not None:
        assert type(eq) == list and len(eq) == 2, "等式约束条件为形如 [A, B] 的列表"
        a1, b1 = matrix(eq[0]), matrix(eq[1])
        a, b = matrix([a, a1]), matrix([b, b1])

    solvers.options['show_progress'] = show_progress
    solvers.options['maxiters'] = 1000
    sol = solvers.cpl(r, func, g, h, A=a, b=b)
    return sol['x']


def min_risk(returns, risk_structure, target_return, base=None, up=None, eq=None, noeq=None):
    """
    给定目标收益，最小化风险
    :param returns: 下一期的股票收益
    :param risk_structure: 风险结构
    :param target_return: 目标收益
    :param base: 基准，可以为None
    :param up: 权重上限
    :param eq: list, [A, B], 其余等式约束条件
    :param noeq: list, [G, h], 其余不等式约束条件
    :return: 最优化的投资组合权重
    """
    assert len(risk_structure) == len(returns), "numbers of companies in risk structure " \
                                                "and returns vector are not the same"
    if base is not None:
        assert len(base) == len(returns), "numbers of companies in  base vector and returns vector are not the same"

    if up is None:
        up = np.ones(len(returns))  # 默认取值为1
    else:
        up = np.ones(len(returns)) * np.asarray(up)  # up 既可以为值也可以为列向量

    p = matrix(np.asarray(risk_structure))
    num = len(returns)
    q = matrix(np.zeros(num))
    base = matrix(np.asarray(base if base is not None else np.zeros(num))) * 1.0

    # 不能卖空
    g1 = matrix(np.diag(np.ones(num) * -1.0))
    h1 = base
    # 权重上限
    g2 = matrix(np.diag(np.ones(num)))
    h2 = matrix(up) - base
    # 目标收益
    g3 = matrix(np.asarray(returns)).T * -1.0
    h3 = matrix(-1.0 * target_return, (1, 1))
    g, h = matrix([g1, g2, g3]), matrix([h1, h2, h3])
    # 权重和为0 或 1
    a = matrix(np.ones(num)).T
    b = matrix(0.0 if sum(base) > 0 else 1.0, (1, 1))

    if noeq is not None:
        assert type(noeq) == list and len(noeq) == 2, "不等式约束条件为形如 [G, H] 的列表"
        g3, h3 = matrix(noeq[0]), matrix(noeq[1])
        g, h = matrix([g, g3]), matrix([h, h3])

    if eq is not None:
        assert type(eq) == list and len(eq) == 2, "等式约束条件为形如 [A, B] 的列表"
        a1, b1 = matrix(eq[0]), matrix(eq[1])
        a, b = matrix([a, a1]), matrix([b, b1])

    solvers.options['show_progress'] = show_progress
    solvers.options['maxiters'] = 1000
    try:
        sol = solvers.qp(p, q, g, h, a, b)
    except ValueError:
        raise ValueError("Error in min_risk():make sure your equation can be solved")
    else:
        return sol['x']

