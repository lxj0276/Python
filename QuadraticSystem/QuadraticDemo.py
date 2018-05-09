import pandas as pd
import numpy as np

class Quadratic():
    def __init__(self, x, returns):
        """
        初始化
        :param x:1-T期，股票池A,B,C...到因子 f1,f2,f3...上的因子暴露
        :param returns:1-T期，股票池A,B,C...的收益率
        """
        self.x = x
        self.returns = returns
        self.hist_factor_returns = list()
        self.predict_factor_returns = list()
        self.predict_stock_returns = list()
        self.predict_factor_loading = pd.DataFrame()
        self.returns_residuals = pd.DataFrame()
        self.risk_structure = pd.DataFrame()

    def grading(self, start, end):
        """
        将因子暴露打分
        :param start: 市场因子的起始索引位置
        :param end: 市场因子的结束索引位置
        """
        rank = self.x.apply(lambda x:x[start:end].rank(method='first'), axis=1)
        self.x.iloc[:,start:end] = rank

    def cal_hist_factor_returns(self):
        """
        计算往期因子收益率
        """
        grouped = self.x.groupby(self.x['date'])
        self.hist_factor_returns = grouped.apply()  # 回归
        self.hist_returns_residuals = None  # 收集残差

    def predict_factor_returns(self):
        """
        计算预期因子收益率
        """
        pass

    def predict_stock_returns(self):
        """
        计算预期股票收益率载荷
        """
        pass

    def cal_risk_structure(self):
        """
        计算市场风险结构矩阵 V
        """
        factor_cov = np.cov(self.hist_factor_returns)  # 计算往期因子收益率的协方差矩阵
        factor_nums = np.alen(factor_cov)
        residuals_cov = np.cov(self.returns_residuals)
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