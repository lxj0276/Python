import pandas as pd
import numpy as np
import statsmodels.api as sm
from functools import reduce
from itertools import product

class Quadratic():
    def __init__(self, x, factor):
        """
        :param x: DataFame, T periods, The factor loadings of companies A,B,C... on factors f1,f2,f3...
        :param factor: List, factors
        """
        self.x = x
        self.names = {'date' : 'Date', 'returns' : 'Return',
                      'company' : 'CompanyCode', 'factor' : factor}

    @staticmethod
    def ewma(x, weight):
        """
        :param x: Series
        :param weight: weight
        :return: ewma result
        """
        return reduce(lambda y, z: (1 - weight) * y + weight * z, x)

    def set_names(self, date=None, returns=None, company=None, factor=None):
        """
        set the column names that will be used in calculation
        :param date: the name of the date column in raw data
        :param returns: the name of the returns column in raw data
        :param company: the name of the company column in raw data
        :param factor: the name of factors columns in raw data
        """
        if date:
            self.names['date'] = date
        if returns:
            self.names['returns'] = returns
        if company:
            self.names['company'] = company
        if factor:
            self.names['factor'] = factor

    def ranking(self):
        """
        rank the market factors in need in the following calculation
        """
        factor = self.names['market_factor']
        rank = self.x.apply(lambda x:x[factor].rank(method='first'), axis=1)
        self.x[factor] = rank

    def hist_residuals(self, factor_returns):
        """
        get history residuals from regression results
        :param factor_returns: DataFrame, factor returns as the regression results
        :return: DataFrame with index as dates and columns as companies
        """
        # group by date
        date, returns, company, factor = self.names.values()
        grouped = self.x.groupby(self.x[date])

        # get residuals respectively
        def f(x, params): return x[returns]-(x[factor]*next(params)).sum(axis=1)
        results_residuals = None
        try:
            g = (list(factor_returns[factor].iloc[i]) for i in range(len(factor_returns)))
            results_residuals = pd.DataFrame([list(f(group, g)) for _, group in grouped])
            results_residuals.columns = self.x[company].unique()
        except StopIteration:
            pass

        return results_residuals

    def hist_factor_returns(self):
        """
        history factor returns using regression
        :return: DataFrame with index as dates and columns as factors
        """
        # divide the DataFrame into T periods
        date, returns, _, factor = self.names.values()
        grouped = self.x.groupby(self.x[date])

        # T OLS on T groups
        def f(x): return sm.OLS(x[returns], x[factor]).fit().params
        results = grouped.apply(f)

        return results

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

    def predict_factor_loadings(self, method, arg=None):
        """
        predict factor loadings
        :param method: method to predict
        :param arg: additional parameter used in prediction
        :return: DataFrame with index as company and columns as factors
        """
        _, _, company, factor = self.names.values()
        if method == 'average':
            predicts = self.x[factor].groupby(self.x[company]).mean()
        elif method == 'ewma':
            predicts = self.x[factor].groupby(self.x[company]).apply(self.ewma, weight=arg)
        else:
            raise ValueError("predict_factor_loadings:undefined method" + method)
        return predicts

    def predict_stock_returns(self, factor_loadings, factor_returns):
        """
        predict stock returns in period T+1
        :param factor_loadings: factor loadings in period T+1
        :param factor_returns: factor returns in period T+1
        :return: stock returns in period T+1
        """
        return factor_loadings.apply(lambda x:x*factor_returns, axis=1).sum(axis=1)

    def risk_structure(self, method, arg=None):
        """
        get the risk structure matrix V
        :param method: method used in prediction
        :param arg: additional parameter in prediction
        """
        companies = self.x[self.names['company']].unique()
        factors = self.x[self.names['factor']].columns

        # history factor returns and the residuals
        hist_factors_returns = self.hist_factor_returns()
        hist_residuals = self.hist_residuals(hist_factors_returns)

        # predict factor loadings
        predict_loadings = self.predict_factor_loadings(method, arg)

        # covariance matrix of history factor returns and residuals
        factor_cov = np.cov(hist_factors_returns)
        residuals_cov = pd.DataFrame(np.cov(hist_residuals), columns=companies, index=companies)
        factor_nums = np.alen(factor_cov)

        h = len(companies)
        risk_structure = pd.DataFrame(np.ones(h**2).reshape(h, h), columns=companies, index = companies)
        # sum
        for i in companies:
            for j in companies:
                l = [predict_loadings.at[i, factors[k1]] *
                     predict_loadings.at[j, factors[k2]] *
                     factor_cov[k1, k2] for k1 in range(factor_nums) for k2 in range(factor_nums)]
                risk_structure.at[i, j] = sum(l) + residuals_cov.at[i, j]

        return risk_structure

if __name__ == '__main__':
    """
    data = pd.read_csv('expo_test.csv')
    print(data.head())
    data.groupby()
    factors = ['PE', 'PB', 'PS', 'ROE', 'ROA']
    quad = Quadratic(data, factors)
    quad.set_names(date='Month')
    
    # data cleaning
    companies = data.groupby(data['Date']).apply(lambda x:pd.Series(x['CompanyCode']))
    s1 = reduce(lambda x, y: set(x).intersection(set(y)), (companies.iloc[i] for i in range(len(companies))))
    filter1 = pd.Series([i in s1 for i in data['CompanyCode']])
    data = data[filter1]

    dates = data.groupby(data['CompanyCode']).apply(lambda x: pd.Series(x['Date']))
    s2 = reduce(lambda x, y: set(x).intersection(set(y)), (dates.iloc[i] for i in range(len(companies))))
    filter2 = pd.Series([i in s2 for i in data['Date']])
    data = data[filter2]
    """
    date = ['2018-01-01', '2018-01-02', '2018-01-03']
    company = ['A', 'B', 'C']
    data = pd.DataFrame(list(product(date, company)), columns=['date', 'company'])
    data[['PE','PB','PS']]= pd.DataFrame(np.arange(27).reshape(9,3))
    factors = ['PE','PB','PS']
    data['return'] = np.arange(9)
    quad = Quadratic(data, factors)

    quad.set_names(date='date', company='company', returns='return')
    print(quad.hist_factor_returns())
    print(quad.predict_factor_returns(quad.hist_factor_returns(), method='ewma', arg=0.5))
    print(quad.hist_residuals(quad.hist_factor_returns()))
    a = quad.predict_factor_loadings(method='average')
    print(quad.predict_stock_returns(quad.predict_factor_loadings(method='average'), quad.predict_factor_returns(quad.hist_factor_returns(), method='ewma', arg=0.5)))
    print(quad.risk_structure(method = 'average'))