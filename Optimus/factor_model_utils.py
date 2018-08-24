import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.filters import hp_filter
from functools import reduce


def ewma(x, weight):
    """
    :param x: Series
    :param weight: 权重
    :return: ewma结果
    """
    return reduce(lambda y, z: (1 - weight) * y + weight * z, x)


def predict_factor_returns(hist_factor_return, method, arg=0.5):
    """
    预测下一期的因子收益
    :param hist_factor_return: DataFrame, 历史收益矩阵
    :param method: str, 预测方法
    :param arg: 预测方法需要的参数
    :return: 下一期的因子收益
    """
    if method == 'average':
        predicts = hist_factor_return.mean()
    elif method == 'ewma':
        predicts = hist_factor_return.apply(ewma, weight=arg)
    elif method == 'hpfilter':
        def f(x):
            _, trend = hp_filter.hpfilter(x, 129600)
            return trend.iloc[-1]

        predicts = hist_factor_return.apply(f)
    else:
        raise ValueError("predict_factor_returns:undefined method：" + method)
    return predicts


def predict_stock_returns(factor_loadings, predict_factor_return):
    """
    预测第T+1期的股票收益
    :param factor_loadings: 第T期的因子暴露
    :param predict_factor_return: 预测的因子收益
    :return: 第T+1期的股票收益
    """
    return factor_loadings.apply(lambda x: x * predict_factor_return, axis=1).sum(axis=1)


def risk_structure(hist_factor_return, factor_loadings, hist_residual=None):
    """
    获取多因子模型中的风险结构
    :param hist_factor_return: 历史的因子收益
    :param factor_loadings: 第T期的因子暴露
    :param hist_residual: 历史残差
    """
    # 历史因子收益以及残差的协方差矩阵
    factor_cov = np.cov(np.asmatrix(hist_factor_return).T)

    # 求和
    assert factor_loadings.shape[1] == factor_cov.shape[0], \
        "factor_loading 和 hist_factor_return 中的因子数量不一致"
    rs = np.dot(np.asmatrix(factor_loadings), factor_cov)
    rs = np.dot(rs, np.asmatrix(factor_loadings).T)

    if hist_residual is not None:
        residuals_cov = np.cov(np.asmatrix(hist_residual))
        diag = np.diag(np.ones(len(hist_residual)))
        residuals_cov = np.where(diag, residuals_cov, diag)
    else:
        residuals_cov = np.zeros_like(rs)

    assert rs.shape == residuals_cov.shape, \
        "应保证hist_residuals中股票的数量与factor_loading中的股票数量一致"
    return rs + residuals_cov


def hist_factor_returns(stock_return, factor, freq):
    """
    历史因子收益序列
    :return: 历史因子收益
    """
    assert len(stock_return) == len(factor) and len(stock_return) == len(freq), "输入参数的长度应相等！"
    factor.index = stock_return.index
    x = pd.concat([stock_return, factor], axis=1)

    # 分为T期
    grouped = x.groupby(freq)

    # 对T期做T次回归
    try:
        def f(x):
            return sm.OLS(x.iloc[:, 0], x.iloc[:, 1:]).fit().params

        results = grouped.apply(f)
    except np.linalg.linalg.LinAlgError:
        raise np.linalg.linalg.LinAlgError("Error in hist_factor_returns: "
                                           "输入的变量可能存在多重共线性而不能进行OLS回归")
    else:
        return results.dropna()
