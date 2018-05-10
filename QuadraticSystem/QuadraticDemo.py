import pandas as pd
import numpy as np
import statsmodels.api as sm


class Quadratic():
    def __init__(self, x, returns, all_factors, market_factors):
        """
        usage: 初始化
        :param x:DataFame, 1-T期，股票池A,B,C...到因子 f1,f2,f3...上的因子暴露
        :param returns:Series, 1-T期，股票池A,B,C...的收益率
        :param all_factors:List
        :param market_factors:List
        """
        self.x = x
        self.x['returns'] = returns
        self.all_factors = all_factors
        self.market_factors = market_factors
        # self.hist_factor_returns = list()
        # self.predict_factor_returns = list()
        self.predict_stock_returns = list()
        self.predict_factor_loading = pd.DataFrame()
        # self.returns_residuals = pd.DataFrame()
        self.risk_structure = pd.DataFrame()

    def grading(self):
        """
        usage: 将因子暴露打分
        """
        rank = self.x.apply(lambda x:x[self.market_factors].rank(method='first'), axis=1)
        self.x[self.market_factors] = rank

    def get_hist_residuals(self, results):
        """
        usage: 计算残差
        :param results: 往期因子收益率
        :return: 残差
        """
        right_cols = results.columns = results.columns + '_h'
        data_merged = pd.merge(self.x, results, left_on='Date', right_index=True)
        grouped = data_merged.groupby(self.x['Date'])
        func = lambda x:x['returns']-(x[self.all_factors]*x[right_cols]).sum(axis=1)  # 计算残差
        results_residuals = grouped.apply(func)

        return results_residuals

    def get_hist_factor_returns(self):
        """
        usage: 计算往期因子收益率
        """
        # 回归得到往期因子收益
        grouped = self.x.groupby(self.x['Date'])
        func = lambda x: sm.OLS(x['returns'], x[self.all_factors]).fit().params  # OLS
        results = grouped.apply(func)

        return results

    def predict_factor_returns(self, factor_returns, method):
        """
        usage: 计算预期因子收益率
        :param factor_returns: DataFrame, 因子历史收益矩阵
        :param method: str, 估计 T+1 期收益的方法
        """
        pass

    def predict_stock_returns(self):
        """
        usage: 计算预期股票收益率
        """
        pass

    def get_risk_structure(self):
        """
        usage: 计算市场风险结构矩阵 V
        """
        #  计算往期因子收益并获取残差矩阵
        hist_fac_ret = self.get_hist_factor_returns()
        hist_res = self.get_hist_residuals(hist_fac_ret)

        # 计算往期因子收益率和股票残差的协方差矩阵
        factor_cov = np.cov(hist_fac_ret)
        factor_nums = np.alen(factor_cov)
        residuals_cov = np.cov(hist_res)

        stocks = self.x['stocks'].unique()
        self.risk_structure = pd.DataFrame(np.ones(len(stocks)*len(stocks)), columns=stocks)
        for i in range(len(stocks)):
            for j in range(len(stocks)):
                l = [self.predict_factor_loading.at[i, k1]*
                     self.predict_factor_loading.at[j, k2]*
                     factor_cov[k1, k2] for k1 in range(factor_nums) for k2 in range(factor_nums)]
                self.risk_structure.at[i, j] = sum(l) + residuals_cov[i, j]


def main():
    pass


if __name__ == '__main__':
    main()