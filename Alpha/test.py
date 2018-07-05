import numpy as np
import pandas as pd
from numpy.linalg import inv
from itertools import combinations, chain


def cal_all_set(num):
    index = [list(combinations(range(num), i)) for i in range(1, num+1)]
    return list(chain(*index))


def regress(y, x):
    n, p = x.shape
    X = np.asmatrix(x)
    Y = np.asmatrix(y).reshape(n, 1)

    # 计算系数以及拟合值
    mat = inv(np.dot(X.T, X))
    b = np.dot(mat, np.dot(X.T, Y))
    fitted = np.dot(X, b)

    # 计算t值
    varhat = (np.dot((Y - fitted).T, (Y - fitted)) / (n - p))
    tscore = [b[i, 0] / (varhat[0, 0] * mat[i, i]) ** 0.5 for i in range(p)]

    # 计算 R2
    r2 = 1 - varhat[0, 0] / np.var(y, ddof=1)

    return r2, b.getA1(), tscore, varhat[0, 0]


def cal_best_subset(nav, factors):
    factors = np.asarray(factors)
    nobs, factor_num = factors.shape

    all_set = cal_all_set(factor_num)
    index = 0
    best_r2 = 0
    for i in range(len(all_set)):
        x = np.column_stack([np.ones(nobs), factors[:, all_set[i]]])
        r2 = regress(nav, x)
        if r2 > best_r2:
            best_r2, index = r2, i

    return all_set[index]


def cal_fund_order(fund_data, base, factors):
    """
    计算基金的rank
    :param fund_data:   基金净值数据
    :param base:        基准净值数据
    :param factors:     因子数据
    :return:            排序结果
    """
    funds = fund_data.columns
    targets = ['alpha', 't-alpha', 'r-square', 'ar']

    result = pd.DataFrame(np.zeros((len(funds), len(targets))), columns=targets, index=funds)
    for f in funds:
        sub = cal_best_subset(fund_data[f], factors)                        # 找到最佳因子集
        x = np.column_stack([np.ones(len(fund_data)), factors[:, sub]])     # 添加beta0
        r2, b, tscore, varhat = regress(fund_data[f], x)                    # 使用最佳因子集得到回归结果
        ar = (0.5 * b[0] / varhat ** 0.5 +                                  # 求评估比率
              0.5 * np.mean(base - fund_data[f]) / np.std(base - fund_data[f], ddof=1))
        result.loc[f] = b[0], tscore[0], r2, ar

    def norm(arr):
        return (arr - np.mean(arr)) / np.std(arr, ddof=1)

    result = result.apply(norm)
    w = [0.25, 0.25, 0.25, 0.25]
    ranked_index = (result * w).sum(axis=1).sortvalues().index
    return result.reindex(ranked_index)


if __name__ == '__main__':
    pass

