import sys
import warnings
from functools import reduce

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.filters import hp_filter
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
                      'industry': 'IndustryName'}
        if True not in {self.__is_const(x[i]) for i in list(factor)}:
            warnings.warn("Warning in Optimus(): Missing one column as the market factor, "
                          "try to add one column as a single-value column like 1.0")

        self.__hfr = None
        self.__hr = None
        self.__pfr = None
        self.__industry = None
        self.__factor_loading = None
        self.__psr = None
        self.__rs = None
        self.__B = None
        self.__te = None
        self.__up = None
        self.__deviate = None

    @staticmethod
    def __is_const(col):
        if len(set(col)) == 1:
            return True
        return False

    @staticmethod
    def __ewma(x, weight):
        """
        :param x: Series
        :param weight: weight
        :return: ewma result
        """
        return reduce(lambda y, z: (1 - weight) * y + weight * z, x)

    @staticmethod
    def __predict_stock_returns(factor_loadings, predict_factor_returns):
        """
        predict stock returns in period T+1
        :param factor_loadings: factor loadings in period T
        :param predict_factor_returns: factor returns in period T+1
        :return: stock returns in period T+1
        """
        return factor_loadings.apply(lambda x: x * predict_factor_returns, axis=1).sum(axis=1)

    @staticmethod
    def __risk_structure(hist_factor_returns, hist_residuals, factor_loadings):
        """
        get the risk structure matrix V
        :param hist_factor_returns: history factor returns
        :param hist_residuals: history residuals
        :param factor_loadings: factor loadings in period T
        """
        # covariance matrix of history factor returns and residuals
        factor_cov = np.cov(np.asmatrix(hist_factor_returns).T)
        residuals_cov = np.cov(np.asmatrix(hist_residuals))
        diag = np.diag(np.ones(len(hist_residuals)))
        residuals_cov = np.where(diag, residuals_cov, diag)

        # sum
        try:
            risk_structure = np.dot(np.asmatrix(factor_loadings), factor_cov)
            risk_structure = np.dot(risk_structure, np.asmatrix(factor_loadings).T)
        except ValueError:
            print("ValueError: risk_structure(): "
                  "factors in factor loadings and history factor returns are not the same")
            sys.exit(1)

        try:
            return risk_structure + residuals_cov
        except ValueError:
            print("ValueError: risk_structure(): "
                  "number of companies in factor loadings is not the same as it in residuals")
            sys.exit(1)

    @staticmethod
    def __max_returns(returns, risk_structure, te, Base, up, industry=None, deviate=None, factor=None, xk=None):
        """
        calculate the best portfolios, maximize returns give the risk sigma2
        :param returns: future stock returns
        :param risk_structure: risk structure
        :param te: Double, upper bound of Tracking Error
        :param Base: Vector, Basic portfolio
        :param up: Double, upper bound of weight
        :param industry: DataFrame, dummy variables for industry for N companies, control industry risk
        :param deviate: Double, deviate bound of industry
        :param factor: DataFrame, future factor loadings, control factor risk
        :param xk: Double, upper bound of factor risk
        :return: optimal portfolio
        """
        assert len(Base) == len(returns), "numbers of companies in  base vector and returns vector are not the same"
        assert len(risk_structure) == len(returns), "numbers of companies in risk structure " \
                                                    "and returns vector are not the same"
        assert len(industry) == len(returns), "numbers of companies in industry dummy matrix " \
                                              "not equals to it in returns vector"

        r, V = matrix(np.asarray(returns))*-1, matrix(np.asarray(risk_structure))
        num = len(returns)
        B = matrix(np.asarray(Base)) * 1.0

        def F(x=None, z=None):
            if x is None:
                return 1, matrix(0.0, (len(r), 1))
            f = x.T * V * x - te**2 / 12
            Df = x.T * (V + V.T)
            if z is None:
                return f, Df
            return f, Df, z[0, 0] * (V + V.T)

        # Basic portfolio Bound
        G1 = matrix(np.diag(np.ones(num) * -1))
        h1 = B
        # upper weight
        G2 = matrix(np.diag(np.ones(num)))
        h2 = matrix(up, (num, 1)) - B
        G, h = matrix([G1, G2]), matrix([h1, h2])
        # sum = 0.0
        A = matrix(np.ones(num)).T
        b = matrix(0.0, (1, 1))

        # factor bound
        if factor is not None:
            G3 = matrix(np.asarray(factor)).T
            h3 = matrix(xk, (len(factor.columns), 1))
            G, h = matrix([G, G3]), matrix([h, h3])

        # hedge industry risk
        if industry is not None:
            m = matrix(np.asarray(industry)).T * 1.0
            c = matrix(deviate, (len(industry.columns), 1))
            if deviate == 0.0:
                A, b = m, c
            elif deviate > 0.0:
                G, h = matrix([matrix([G, m]),-m]), matrix([matrix([h, c]), c])

        # solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 1000
        sol = solvers.cpl(r, F, G, h, A=A, b=b)
        return sol['x']

    @staticmethod
    def __min_risk(returns, risk_structure, target_return, up):
        """
        Given the target-return, minimize risk
        :param returns: stock returns in T+1
        :param risk_structure: risk structure
        :param target_return: target return
        :param up: upper bound of weight
        :return: portfolio weights
        """
        assert len(risk_structure) == len(returns), "numbers of companies in risk structure " \
                                                    "and returns vector are not the same"

        P = matrix(np.asarray(risk_structure))
        num = len(returns)
        q = matrix(np.zeros(num))

        # upper weight
        G1 = matrix(np.diag(np.ones(num)))
        h1 = matrix(up, (num, 1))
        # target return
        G2 = matrix(np.asarray(returns)).T * -1
        h2 = matrix(-1*target_return, (1,1))
        G, h = matrix([G1, G2]), matrix([h1, h2])
        # sum = 1.0
        A = matrix(np.ones(num)).T
        b = matrix(1.0, (1, 1))

        # solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 1000
        try:
            sol = solvers.qp(P, q, G, h, A, b)
        except ValueError as e:
            print("Error in min_risk():make sure your equation can be solved")
            sys.exit(1)
        else:
            return sol['x']

    def rank_factors(self):
        """
        rank the market factors in need in the following calculation
        """
        factor = self.names['factor']
        rank = self.x[factor].rank()
        self.x[factor] = rank
        mean = self.x[factor].mean()
        std = self.x[factor].std()
        self.x[factor] = self.x[factor].apply(lambda x: (x - mean) / std, axis=1)

    def __hist_factor_returns(self):
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

    def __hist_residuals(self, factor_returns):
        """
        get history residuals from regression results
        :param factor_returns: DataFrame, factor returns as the regression results
        :return: DataFrame with index as dates and columns as companies
        """
        # group by date
        freq, returns, company, factor = list(self.names.values())[:4]
        periods = self.x[freq].unique()

        g = (list(factor_returns[factor].iloc[i]) for i in range(len(factor_returns)))

        def f(x, params):
            return x[returns] - (x[factor] * next(params)).sum(axis=1)

        results_residuals = pd.DataFrame()
        index = pd.Index([])
        for period in periods[:-1]:
            slice = self.x[self.x[freq] == period]
            col = f(slice, g)
            col.index = slice[company]
            index = index.union(slice[company])
            col = col.reindex(index)
            results_residuals = results_residuals.reindex(index)
            results_residuals[period] = col

        results_residuals = results_residuals.reindex(self.x[self.x[freq] == periods[-1]][company])
        return results_residuals.apply(lambda x: x.fillna(x.mean()), axis=1)
    
    def __predict_factor_returns(self, factor_returns, method, arg=0.5):
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
            predicts = factor_returns.apply(self.__ewma, weight=arg)
        elif method == 'hpfilter':
            def f(x):
                _, trend =  hp_filter.hpfilter(x, 129600)
                return trend.iloc[-1]
            predicts = factor_returns.apply(f)
        else:
            raise ValueError("predict_factor_returns:undefined method：" + method)
        return predicts

    def __get_factor_loading_t(self):
        """
        get period t's factor loading
        :return: factor loading
        """
        # filter data at period T
        freq, factor, company = self.names['freq'], self.names['factor'], self.names['company']
        periods = self.x[freq].unique()
        data_t = self.x[self.x[freq] == periods[-1]]

        # return factor loading
        factor_loading = data_t[factor]
        factor_loading.index = data_t[company]
        return factor_loading

    def __get_industry_dummy(self):
        """
        get industry dummies
        :return: DataFrame
        """
        # filter data at period T
        freq, industry_name, company = self.names['freq'], self.names['industry'], self.names['company']
        freqs = self.x[freq].unique()
        data_t = self.x[self.x[freq] == freqs[-1]]
        names = data_t[industry_name].unique()

        # construct dummy matrix
        industry = pd.DataFrame()
        for name in names:
            industry[name] = data_t[industry_name].map(lambda x: 1 if x == name else 0)
        industry.index = data_t[company]

        return industry

    def init(self):
        # get history factor returns
        self.__hfr = self.__hist_factor_returns()
        self.__hr = self.__hist_residuals(self.__hfr)

        # get factor loadings at period T
        self.__factor_loading = self.__get_factor_loading_t()

        # predict factor returns at period T+1
        self.__pfr = self.__predict_factor_returns(self.__hfr, 'hpfilter')

        # risk structure
        self.__industry = self.__get_industry_dummy()

        self.__psr = self.__predict_stock_returns(self.__factor_loading, self.__pfr)
        self.__rs = self.__risk_structure(self.__hfr, self.__hr, self.__factor_loading)

    def optimize_returns(self):
        return self.__max_returns(self.__psr, self.__rs, self.__te, self.__B, self.__up, self.__industry, self.__deviate)

    def get_components(self):
        freq, factor, company = self.names['freq'], self.names['factor'], self.names['company']
        periods = self.x[freq].unique()
        data_t = self.x[self.x[freq] == periods[-1]]

        return list(data_t[company])

    def set_names(self, freq=None, returns=None, company=None, factor=None, industry=None):
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
            self.names['factor'] = list(factor)
        if industry:
            self.names['industry'] = industry

    def set_hr(self, hr):
        self.__hr = hr

    def set_industry(self, industry):
        self.__industry = industry

    def set_factor_loading(self, factor_loading):
        self.__factor_loading = factor_loading

    def set_base(self, base):
        self.__B = base

    def set_te(self, te):
        self.__te = te

    def set_up(self, up):
        self.__up = up

    def set_deviate(self, deviate):
        self.__deviate = deviate

    def set_predict_method(self, method):
        self.__pfr = self.__predict_factor_returns(self.__hfr, method)

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

    op.init()

    B = np.ones(288)/288
    op.set_base(B)
    op.set_te(0.07)
    op.set_up(0.01)
    op.set_deviate(0.0)
    print(op.optimize_returns())
