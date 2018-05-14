import pandas as pd
import numpy as np
import statsmodels.api as sm
from functools import reduce
from cvxopt import matrix, solvers


class Optimus():
    def __init__(self, x, factor):
        """
        :param x: DataFame, T periods, The factor loadings of companies A,B,C... on factors f1,f2,f3...
        :param factor: List, factors
        """
        self.x = x
        self.names = {'date': 'Date', 'returns': 'Return',
                      'company': 'CompanyCode', 'factor': factor,
                      'industry': [i for i in factor if self.is_industry(x[i])],
                      'market': [i for i in factor if not self.is_industry(x[i])]}

    @staticmethod
    def is_industry(col):
        for i in col:
            if i not in (1,0):
                return False
        return True

    @staticmethod
    def ewma(x, weight):
        """
        :param x: Series
        :param weight: weight
        :return: ewma result
        """
        return reduce(lambda y, z: (1 - weight) * y + weight * z, x)

    @staticmethod
    def predict_stock_returns(predict_factor_loadings, predict_factor_returns):
        """
        predict stock returns in period T+1
        :param predict_factor_loadings: factor loadings in period T+1
        :param predict_factor_returns: factor returns in period T+1
        :return: stock returns in period T+1
        """
        return predict_factor_loadings.apply(lambda x: x * predict_factor_returns, axis=1).sum(axis=1)

    @staticmethod
    def max_returns(returns, risk_structure, sigma2, B, up, industry=None, factor=None, xk=None):
        """
        calculate the best portfolios, maximize returns give the risk sigma2
        :param returns: future stock returns
        :param risk_structure: risk structure
        :param sigma2: Double, risk
        :param B: Vector, Basic portfolio
        :param up: Double, upper bound of weight
        :param industry: DataFrame, dummy variables for industry for N companies, control industry risk if not none
        :param factor: DataFrame, future factor loadings, control factor risk if not none
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

        # sum = 1
        A = matrix(np.ones(num)).T
        b = matrix(0.0, (1,1))

        # hedge industry risk
        if industry is not None:
            A1 = matrix(np.asarray(industry)).T
            b1 = matrix(0.0, (len(industry.columns), 1))
            A, b = matrix([A, A1]), matrix([b, b1])

        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 1000
        sol = solvers.cpl(r, F, G, h, A=A, b=b)
        return sol['x']

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
            self.names['industry'] = [i for i in factor if self.is_industry(self.x[i])]
            self.names['market'] = [i for i in factor if not self.is_industry(self.x[i])]

    def rank_factors(self):
        """
        rank the market factors in need in the following calculation
        """
        factor = self.names['market']
        rank = self.x[factor].rank(method='dense')
        self.x[factor] = rank
        mean = self.x[factor].mean()
        std = self.x[factor].std()
        self.x[factor] = self.x[factor].apply(lambda x:(x-mean)/std, axis=1)

    def hist_residuals(self, factor_returns):
        """
        get history residuals from regression results
        :param factor_returns: DataFrame, factor returns as the regression results
        :return: DataFrame with index as dates and columns as companies
        """
        # group by date
        date, returns, company, factor, _, _ = self.names.values()
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

        return results_residuals.applymap(lambda x:0.0 if x<0.01 else x)

    def hist_factor_returns(self):
        """
        history factor returns using regression
        :return: DataFrame with index as dates and columns as factors
        """
        # divide the DataFrame into T periods
        date, returns, _, factor, _, _ = self.names.values()
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
        _, _, company, factor, _, _ = self.names.values()
        if method == 'average':
            predicts = self.x[factor].groupby(self.x[company]).mean()
        elif method == 'ewma':
            predicts = self.x[factor].groupby(self.x[company]).apply(lambda x, a: x.ewm(alpha=a).mean()[-1:], a=arg)
            predicts.index = self.x[company].unique()
        else:
            raise ValueError("predict_factor_loadings:undefined method" + method)
        return predicts

    def risk_structure(self, hist_factor_returns, hist_residuals, predict_factor_loadings):
        """
        get the risk structure matrix V
        :param hist_factor_returns: history factor returns
        :param hist_residuals: history residuals
        :param predict_factor_loadings: predict factor loadings
        """
        companies = self.x[self.names['company']].unique()
        factors = self.x[self.names['factor']].columns

        # covariance matrix of history factor returns and residuals
        factor_cov = np.cov(hist_factor_returns)
        residuals_cov = pd.DataFrame(np.cov(hist_residuals), columns=companies, index=companies)
        factor_nums = np.alen(factor_cov)

        h = len(companies)
        risk_structure = pd.DataFrame(np.ones(h**2).reshape(h, h), columns=companies, index= companies)
        # sum
        for i in companies:
            for j in companies:
                s =sum([predict_factor_loadings.at[i, factors[k1]] *
                        predict_factor_loadings.at[j, factors[k2]] *
                        factor_cov[k1, k2] for k1 in range(factor_nums) for k2 in range(factor_nums)])
                risk_structure.at[i, j] = s + residuals_cov.at[i, j]

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

    from itertools import product
    data = pd.DataFrame(list(product(date, company)), columns=['date', 'company'])
    data[['PE','PB','PS']]= pd.DataFrame(np.arange(27).reshape(9,3))
    factors = ['PE','PB','PS']
    data['return'] = np.arange(9)
    op = Optimus(data, factors)
    op.rank_factors()
    print(op.x)
    op.set_names(date='date', company='company', returns='return')
    hfr = op.hist_factor_returns()
    print("hfr:")
    print(hfr)
    pfr = op.predict_factor_returns(hfr, method='ewma', arg=0.5)
    print("pfr:")
    print(pfr)
    hr = op.hist_residuals(hfr)
    print("hr:")
    print(hr)
    pfl = op.predict_factor_loadings(method="ewma", arg=0.5)
    print("pfl:")
    print(pfl)
    psr = op.predict_stock_returns(pfl, pfr)
    print("psr:")
    print(psr)
    rs = op.risk_structure(hist_factor_returns=hfr, hist_residuals=hr, predict_factor_loadings=pfl)
    print("rs:")
    print(rs)
    B = np.ones(3)/3
    sol = op.max_returns(psr, rs, 10, B, 0.5)
    print("sol:")
    print(sol)