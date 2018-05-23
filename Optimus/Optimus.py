import sys
import warnings
from functools import reduce

import pandas as pd
import numpy as np
import statsmodels.api as sm
from cvxopt import matrix, solvers
# to rightly install cvxopt see:https://blog.csdn.net/qq_32106517/article/details/78746517

class Optimus():
    def __init__(self, x, factor):
        """
        :param x: DataFame, T periods, The factor loadings of companies A,B,C... on factors f1,f2,f3...
        :param factor: List, factors
        """
        self.x = pd.DataFrame(x)
        self.names = {'freq': 'Date', 'returns': 'Return',
                      'company': 'CompanyCode', 'factor': list(factor),
                      'industry': [i for i in factor if self.is_industry(x[i])],
                      'style': [i for i in factor if not self.is_industry(x[i])]}
        if True not in {self.is_const(x[i]) for i in list(factor)}:
            warnings.warn("Warning in Optimus(): Missing one column as the market factor, "
                          "try to add one column as a single-value column like 1.0")

    @staticmethod
    def is_const(col):
        if len(set(col)) == 1:
            return True
        return False

    @staticmethod
    def is_industry(col):
        if len(set(col)) <= 2:
            return True
        return False

    @staticmethod
    def ewma(x, weight):
        """
        :param x: Series
        :param weight: weight
        :return: ewma result
        """
        return reduce(lambda y, z: (1 - weight) * y + weight * z, x)

    @staticmethod
    def predict_stock_returns(factor_loadings, predict_factor_returns):
        """
        predict stock returns in period T+1
        :param factor_loadings: factor loadings in period T
        :param predict_factor_returns: factor returns in period T+1
        :return: stock returns in period T+1
        """
        return factor_loadings.apply(lambda x: x * predict_factor_returns, axis=1).sum(axis=1)

    @staticmethod
    def risk_structure(hist_factor_returns, factor_loadings):
        """
        get the risk structure matrix V
        :param hist_factor_returns: history factor returns
        :param factor_loadings: factor loadings in period T
        """
        # covariance matrix of history factor returns and residuals
        factor_cov = np.cov(np.asmatrix(hist_factor_returns).T)
        # sum
        risk_structure = np.dot(np.asmatrix(factor_loadings), factor_cov)
        risk_structure = np.dot(risk_structure, np.asmatrix(factor_loadings).T)
        return risk_structure

    @staticmethod
    def max_returns(returns, risk_structure, sigma2, B, up, industry=None, factor=None, xk=None):
        """
        calculate the best portfolios, maximize returns give the risk sigma2
        :param returns: future stock returns
        :param risk_structure: risk structure
        :param sigma2: Double, upper bound of risk
        :param B: Vector, Basic portfolio
        :param up: Double, upper bound of weight
        :param industry: DataFrame, dummy variables for industry for N companies, control industry risk
        :param factor: DataFrame, future factor loadings, control factor risk
        :param xk: Double, upper bound of factor risk
        :return: optimal portfolio
        """
        r, V = matrix(np.asarray(returns)) * -1, matrix(np.asarray(risk_structure))
        num = len(returns)

        def F(x=None, z=None):
            if x is None:
                return 1, matrix(0.0, (len(r), 1))
            f = x.T * V * x - sigma2
            Df = x.T * (V + V.T)
            if z is None:
                return f, Df
            return f, Df, z[0, 0] * (V + V.T)

        # Basic portfolio Bound
        G1 = matrix(np.diag(np.ones(num) * -1))
        h1 = matrix(list(B))
        # upper weight
        G2 = matrix(np.diag(np.ones(num)))
        h2 = matrix(up, (num, 1))
        G, h = matrix([G1, G2]), matrix([h1, h2])

        # factor bound
        if factor is not None:
            G3 = matrix(np.asarray(factor)).T
            h3 = matrix(xk, (len(factor.columns), 1))
            G, h = matrix([G, G3]), matrix([h, h3])

        # sum = 0.0
        A = matrix(np.ones(num)).T
        b = matrix(0.0, (1, 1))

        # hedge industry risk
        if industry is not None:
            A1 = matrix(np.asarray(industry)).T
            b1 = matrix(0.0, (len(industry.columns), 1))
            A, b = matrix([A, A1]), matrix([b, b1])

        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 1000
        sol = solvers.cpl(r, F, G, h, A=A, b=b)
        return sol['x']

    def set_names(self, freq=None, returns=None, company=None, factor=None):
        """
        set the column names that will be used in calculation
        :param freq: the name of the time column
        :param returns: the name of the returns column
        :param company: the name of the company column
        :param factor: the name of factor columns
        """
        if freq:
            self.names['freq'] = freq
        if returns:
            self.names['returns'] = returns
        if company:
            self.names['company'] = company
        if factor:
            self.names['industry'] = [i for i in factor if self.is_industry(self.x[i])]
            self.names['style'] = [i for i in factor if not self.is_industry(self.x[i])]
            self.names['factor'] = list(factor)

    def rank_factors(self):
        """
        rank the market factors in need in the following calculation
        """
        factor = self.names['style']
        rank = self.x[factor].rank()
        self.x[factor] = rank
        mean = self.x[factor].mean()
        std = self.x[factor].std()
        self.x[factor] = self.x[factor].apply(lambda x: (x - mean) / std, axis=1)

    def hist_factor_returns(self):
        """
        history factor returns using regression
        :return: DataFrame with index as dates and columns as factors
        """
        # divide the DataFrame into T periods
        freq, returns, _, factor = list(self.names.values())[:4]
        grouped = self.x.groupby(self.x[freq])

        # T OLS on T groups
        try:
            def f(x): return sm.OLS(x[returns], x[factor]).fit().params
            results = grouped.apply(f)
        except np.linalg.linalg.LinAlgError:
            print("Error in hist_factor_returns: Check if the variables are suitable in OLS")
            sys.exit(1)
        else:
            return results.dropna()

    def predict_factor_returns(self, factor_returns, method, arg=None):
        """
        predict future factor returns in period T+1
        :param factor_returns: DataFrame, the matrix of history factor returns
        :param method: str, the method to predict
        :param arg: additional parameter used in prediction
        :return: Series with index as factors
        """
        if method == 'average':
            predicts = factor_returns.mean()
        elif method == 'ewma':
            predicts = factor_returns.apply(self.ewma, weight=arg)
        else:
            raise ValueError("predict_factor_returns:undefined method：" + method)
        return predicts


if __name__ == '__main__':
    data = pd.read_csv('expo_test.csv')

    factors = list(data.columns.drop(['Ticker', 'CompanyCode', 'TickerName', 'SecuCode', 'IndustryName',
                                      'CategoryName', 'Date', 'Month', 'Return', 'PCF']))

    # remove redundant variables
    factors.remove('AnalystROEAdj')
    factors.remove('FreeCashFlow')

    # create instance
    op = Optimus(data, factors)
    op.set_names(freq='Month')
    print(op.names['factor'])

    # get history factor returns
    hfr = op.hist_factor_returns()
    print(hfr)

    # get factor loadings at period T
    months = data['Month'].unique()
    factors_T = data[data['Month'] == months[-1]]
    factor_loading = factors_T[factors]
    factor_loading.index = factors_T['CompanyCode']
    print(factor_loading)

    # predict factor returns at period T+1
    pfr = op.predict_factor_returns(hfr, 'ewma', 0.5)
    print(pfr)

    # predict stock returns
    psr = op.predict_stock_returns(factor_loading, pfr)
    print(psr)

    # get risk structure
    rs = op.risk_structure(hfr, factor_loading)
    print(rs)

    # construct risk sequence
    B = np.ones(288)/288
    sigma = np.arange(1, 17) * 0.01
    sigma = np.append(np.arange(1, 10) * 0.001, sigma)
    sigma = np.append(np.arange(1, 10) * 0.0001, sigma)
    print(sigma)

    # optimize
    r = [sum(np.array(psr) * list(op.max_returns(psr, rs, i, B, 0.05))) for i in sigma]
    print(r)

    # plot
    import matplotlib.pyplot as plt
    s = pd.Series(r, index=sigma)
    s.plot()
    plt.show()
