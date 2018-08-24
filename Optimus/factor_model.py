import warnings
import pandas as pd
import numpy as np

from factor_model_utils import predict_factor_returns, predict_stock_returns, risk_structure, hist_factor_returns
from optimus import min_risk, max_returns


class Context:

    class Model:
        def __init__(self, psr, rs):
            """
            :param psr: 股票收益
            :param rs:  风险结构
            """
            self.psr = psr
            self.rs = rs

        def max_returns(self, risk, b=None, up=1.0, noeq=None, eq=None):
            """
            给定风险或年化跟踪误差最大化组合收益
            :param risk: 风险或年化跟踪误差
            :param b: 组合基准
            :param up: 个股上限
            :param noeq: 其他不等条件
            :param eq: 其他相等条件
            :return: 最优化的投资组合
            """
            return max_returns(self.psr, self.rs, risk, b, up, noeq, eq)

        def min_risk(self, target_return, b=None, up=1.0, noeq=None, eq=None):
            """
            给定组合目标收益，最小化风险
            :param target_return: 目标组合收益
            :param b: 基准组合
            :param up: 个股上限
            :param noeq: 其他不等条件
            :param eq: 其他相等条件
            :return: 最优化的投资组合
            """
            return min_risk(self.psr, self.rs, target_return, b, up, noeq, eq)

        def print(self):
            print('预测股票收益\n', self.psr)
            print('风险结构\n', self.rs)

    def __init__(self, stock_return, factor, period='m'):
        """
        :param stock_return: Series, 一列代表股票收益率的面板数据
        :param factor: Series 或 DataFrame, 代表因子载荷的面板数据
        :param period:
        """
        assert type(stock_return.index) == pd.DatetimeIndex, "stock_return 的索引应为日期索引 (DatetimeIndex)"
        assert type(factor.index) == pd.DatetimeIndex, "factor 的索引应为日期索引 (DatetimeIndex)"
        assert len(factor) == len(stock_return), "factor 与 stock_return 的长度应相等"
        assert factor.index[0].year != 1970 and stock_return.index[0].year != 1970, "请确认输入数据的索引正确！"

        stock_return.index = factor.index
        self.stock_return = stock_return
        self.factor = factor
        if period.lower() == 'm':
            self.period = factor.index.year * 100 + factor.index.month
        else:
            raise Exception("没有定义的period：{}".format(period))

        self.period = factor.index.month
        if True not in {self.__is_const(self.factor[i]) for i in self.factor.columns}:
            warnings.warn("Warning in Optimus(): Missing one column as the market factor, "
                          "try to add one column as a single-value column like 1.0")

    @staticmethod
    def __is_const(col):
        if len(set(col)) == 1:
            return True
        return False

    def create_factor_model(self, factor_loading):
        """
        创建多因子模型
        :param factor_loading: 用于预测下一期资产收益的因子暴露
        """
        # 获取历史因子收益
        hfr = hist_factor_returns(self.stock_return, self.factor, self.period)

        # 预测下一期的因子收益
        pfr = predict_factor_returns(hfr, 'hpfilter')

        # 结合当期因子暴露计算风险结构
        psr = predict_stock_returns(factor_loading, pfr)
        rs = risk_structure(hfr, factor_loading)
        return self.Model(psr, rs)


if __name__ == '__main__':
    data = pd.read_csv('data/pool.csv')
    from dateutil.parser import parse
    data.index = pd.Index([parse(str(i)) for i in data['Date']])
    data.dropna(inplace=True)
    data.drop(['Index_y', 'Index_x'], axis=1, inplace=True)

    # 选取用于回归的因子
    factors = list(data.columns.drop(
        ['Ticker', 'CompanyCode', 'TickerName', 'SecuCode', 'IndustryName', 'CategoryName', 'Date', 'Month', 'Return',
         'PCF']))

    # remove redundant variables
    factors.remove('AnalystROEAdj')
    factors.remove('FreeCashFlow')

    # 所有日期列表
    date_list = list(data['Date'].unique())

    right = 0
    false = 0
    for i in range(5, len(date_list)-1):
        # 选取最近一期数据
        dd = data[data['Date'] < date_list[i]]
        # 选取最新的factor loading
        latest_factor_loading = data.loc[data['Date'] == date_list[i+1], factors]

        # 创建模型
        context = Context(dd['Return'], dd[factors])
        model = context.create_factor_model(latest_factor_loading)

        # 计算股票数量，并将基准设置为等权
        stock_num = latest_factor_loading.shape[0]
        B = np.ones(stock_num) / stock_num

        try:
            # 最优化
            print(model.min_risk(0.1, B, 0.01))
            right += 1
        except ValueError:
            try:
                print(model.min_risk(0.01, B, 0.01))
                right += 1
            except ValueError:
                false += 1

    print(right)
    print(false)
