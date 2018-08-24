import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.filters import hp_filter
import statsmodels.api as sm

from optimus import min_risk, max_returns, ewma


class FactorModel:
    def __init__(self, x=None, factor=None):
        """
        :param x: DataFame, 用于做多因子模型的数据
        :param factor: List, 多因子模型设计的因子，需要没有NA值，并排除多重共线性
        """
        self.x = pd.DataFrame(x)
        self.factor = None if factor is None else list(factor)
        self.names = {
            'freq': 'Month',
            'returns': 'Return',
            'company': 'CompanyCode',
            'industry': 'IndustryName'
        }
        if factor is not None:
            if True not in {self.__is_const(x[i]) for i in list(factor)}:
                warnings.warn("Warning in Optimus(): Missing one column as the market factor, "
                              "try to add one column as a single-value column like 1.0")

        self.hfr = None
        self.hr = None
        self.pfr = None
        self.factor_loading = None
        self.psr = None
        self.rs = None

    @staticmethod
    def __is_const(col):
        if len(set(col)) == 1:
            return True
        return False

    @staticmethod
    def predict_factor_returns(factor_returns, method, arg=0.5):
        """
        预测下一期的因子收益
        :param factor_returns: DataFrame, 历史收益矩阵
        :param method: str, 预测方法
        :param arg: 预测方法需要的参数
        :return: 下一期的因子收益
        """
        if method == 'average':
            predicts = factor_returns.mean()
        elif method == 'ewma':
            predicts = factor_returns.apply(ewma, weight=arg)
        elif method == 'hpfilter':
            def f(x):
                _, trend = hp_filter.hpfilter(x, 129600)
                return trend.iloc[-1]
            predicts = factor_returns.apply(f)
        else:
            raise ValueError("predict_factor_returns:undefined method：" + method)
        return predicts

    @staticmethod
    def predict_stock_returns(factor_loadings, predict_factor_returns):
        """
        预测第T+1期的股票收益
        :param factor_loadings: 第T期的因子暴露
        :param predict_factor_returns: 预测的因子收益
        :return: 第T+1期的股票收益
        """
        return factor_loadings.apply(lambda x: x * predict_factor_returns, axis=1).sum(axis=1)

    @staticmethod
    def risk_structure(hist_factor_returns, hist_residuals, factor_loadings):
        """
        获取多因子模型中的风险结构
        :param hist_factor_returns: 历史的因子收益
        :param hist_residuals: 历史残差
        :param factor_loadings: 第T期的因子暴露
        """
        # 历史因子收益以及残差的协方差矩阵
        factor_cov = np.cov(np.asmatrix(hist_factor_returns).T)
        residuals_cov = np.cov(np.asmatrix(hist_residuals))
        diag = np.diag(np.ones(len(hist_residuals)))
        residuals_cov = np.where(diag, residuals_cov, diag)

        # 求和
        try:
            risk_structure = np.dot(np.asmatrix(factor_loadings), factor_cov)
            risk_structure = np.dot(risk_structure, np.asmatrix(factor_loadings).T)
        except ValueError:
            raise ValueError("risk_structure(): "
                             "factors in factor loadings and history factor returns are not the same")

        try:
            return risk_structure + residuals_cov
        except ValueError:
            raise ValueError("ValueError: risk_structure(): "
                             "number of companies in factor loadings is not the same as it in residuals")

    def hist_factor_returns(self):
        """
        历史因子收益序列
        :return: 历史因子收益
        """
        freq, returns = list(self.names.values())[:2]
        # 分为T期
        grouped = self.x.groupby(self.x[freq])

        # 对T期做T次回归
        try:
            def f(x):
                return sm.OLS(x[returns], x[self.factor]).fit().params
            results = grouped.apply(f)
        except np.linalg.linalg.LinAlgError:
            raise np.linalg.linalg.LinAlgError("Error in hist_factor_returns: "
                                               "Check if the variables are suitable in OLS")
        else:
            return results.dropna()

    def hist_residuals(self, factor_returns):
        """
        获取历史残差
        :param factor_returns: DataFrame,历史因子收益
        :return: 残差
        """
        # 分组
        freq, returns, company = list(self.names.values())[:3]
        periods = self.x[freq].unique()

        g = (list(factor_returns[self.factor].iloc[i]) for i in range(len(factor_returns)))

        def f(x, params):
            return x[returns] - (x[self.factor] * next(params)).sum(axis=1)

        results_residuals = pd.DataFrame()
        index = pd.Index([])
        for period in periods[:-1]:
            group = self.x[self.x[freq] == period]
            col = f(group, g)
            col.index = group[company]
            index = index.union(group[company])
            col = col.reindex(index)
            results_residuals = results_residuals.reindex(index)
            results_residuals[period] = col

        results_residuals = results_residuals.reindex(self.x[self.x[freq] == periods[-1]][company])
        return results_residuals.apply(lambda x: x.fillna(x.mean()))

    def get_factor_loading_t(self):
        """
        获取当期因子暴露
        :return: 因子暴露
        """
        freq, _, company = list(self.names.values())[:3]
        # 获取当期数据
        periods = self.x[freq].unique()
        data_t = self.x[self.x[freq] == periods[-1]]

        # 返回因子暴露
        factor_loading = data_t[self.factor]
        factor_loading.index = data_t[company]
        return factor_loading

    def create_factor_model(self):
        """
        创建多因子模型
        """
        # 获取历史因子收益及残差
        self.hfr = self.hist_factor_returns()
        self.hr = self.hist_residuals(self.hfr)

        # 获取当期因子暴露
        self.factor_loading = self.get_factor_loading_t()

        # 预测当期的因子收益
        self.pfr = self.predict_factor_returns(self.hfr, 'hpfilter')

        # 风险结构
        self.psr = self.predict_stock_returns(self.factor_loading, self.pfr)
        self.rs = self.risk_structure(self.hfr, self.hr, self.factor_loading)

    def max_returns(self, risk, b=None, up=1.0, industry=None, deviate=None, returns=None, rs=None):
        """
        给定风险或年化跟踪误差最大化组合收益
        :param risk: 风险或年化跟踪误差
        :param b: 组合基准
        :param up: 个股上限
        :param industry: 行业哑变量
        :param deviate: 行业偏离
        :param returns: 预期个股收益
        :param rs: 风险结构
        :return: 最优化的投资组合
        """
        if returns is None:
            returns = self.psr
        if rs is None:
            rs = self.rs

        return max_returns(returns, rs, risk, b, up, industry, deviate)

    def min_risk(self, target_return, b=None, up=1.0, industry=None, deviate=None, returns=None, rs=None):
        """
        给定组合目标收益，最小化风险
        :param target_return: 目标组合收益
        :param b: 基准组合
        :param up: 个股上限
        :param industry: 行业哑变量
        :param deviate: 行业偏离
        :param returns: 预期个股收益
        :param rs: 风险结构
        :return: 最优化的投资组合
        """
        if returns is None:
            returns = self.psr
        if rs is None:
            rs = self.rs

        return min_risk(returns, rs, target_return, b, up, industry, deviate)

    def get_industry_dummy(self):
        """
        获取行业哑变量矩阵
        :return: DataFrame
        """
        freq, _, company, industry_name = list(self.names.values())
        # filter data at period T
        freqs = self.x[freq].unique()
        data_t = self.x[self.x[freq] == freqs[-1]]
        names = data_t[industry_name].unique()

        # construct dummy matrix
        industry = pd.DataFrame()
        for name in names:
            industry[name] = data_t[industry_name].map(lambda x: 1 if x == name else 0)
        industry.index = data_t[company]

        return industry

    def get_components(self):
        """
        获取多因子模型中默认的股票成分，根据最后一期可用的因子暴露的成分股获得
        :return:
        """
        freq, _, company = list(self.names.values())[:3]
        periods = self.x[freq].unique()
        data_t = self.x[self.x[freq] == periods[-1]]

        return list(data_t[company])

    def set_names(self, freq=None, returns=None, company=None, industry=None, factor=None):
        """
        设置计算中用到的列名
        :param freq: 用到的时间列名
        :param returns: 收益列名
        :param company: 公司或股票列名
        :param industry: 行业列名
        :param factor: 因子列名
        """
        if freq:
            assert self.x.columns.contains(freq), "不存在列:" + freq
            self.names['freq'] = freq
        if returns:
            assert self.x.columns.contains(returns), "不存在列:" + returns
            self.names['returns'] = returns
        if company:
            assert self.x.columns.contains(company), "不存在列:" + company
            self.names['company'] = company
        if industry:
            assert self.x.columns.contains(industry), "不存在列:" + industry
            self.names['industry'] = industry
        if factor:
            self.factor = list(factor)

    def set_hr(self, hr):
        self.hr = hr

    def set_factor_loading(self, factor_loading):
        self.factor_loading = factor_loading

    def set_returns(self, returns):
        self.psr = returns

    def set_risk_structure(self, rs):
        self.rs = rs

    def set_predict_method(self, method, arg=None):
        self.pfr = self.predict_factor_returns(self.hfr, method, arg)
        self.psr = self.predict_stock_returns(self.factor_loading, self.pfr)
        self.rs = self.risk_structure(self.hfr, self.hr, self.factor_loading)

    def print_private(self):
        print('历史因子收益矩阵\n', self.hfr)
        print('残差矩阵\n', self.hr)
        print('预测因子收益\n', self.pfr)
        print('因子载荷\n', self.factor_loading)
        print('预测股票收益\n', self.psr)
        print('风险结构\n', self.rs)


if __name__ == '__main__':
    data = pd.read_csv('data/pool.csv')
    data.dropna(inplace=True)
    data.drop(['Index_y', 'Index_x'], axis=1, inplace=True)
    factors = list(data.columns.drop(
        ['Ticker', 'CompanyCode', 'TickerName', 'SecuCode', 'IndustryName', 'CategoryName', 'Date', 'Month', 'Return',
         'PCF']))
    # remove redundant variables
    factors.remove('AnalystROEAdj')
    factors.remove('FreeCashFlow')

    # create instance
    date_list = list(data['Date'].unique())

    right = 0
    false = 0
    for i in date_list[5:]:
        dd = data[data['Date'] < i]
        fm = FactorModel(dd, factors)
        ll = len(fm.get_components())
        fm.set_names(freq='Month')
        fm.create_factor_model()
        B = np.ones(ll) / ll
        industry = fm.get_industry_dummy()
        try:
            print(fm.min_risk(0.1, B, 0.01, industry, 0.01))
            right += 1
        except ValueError:
            try:
                print(fm.min_risk(0.01, B, 0.01, industry, 0.01))
                right += 1
            except ValueError:
                false += 1

    print(right)
    print(false)
