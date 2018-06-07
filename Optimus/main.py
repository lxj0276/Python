import pandas as pd
import numpy as np
import statsmodels.api as sm
from cvxopt import matrix
from Optimus import Optimus


def random_stocks(hs300, industry=None, index=None):
    if industry is None:
        return hs300.sample(n=100)
    data = hs300.groupby(industry).apply(lambda x: x.sample(round(len(x)/3)))
    data.index = data[index]
    return data


def sample(merged, hr, factor_loading, industry, industry_name=None, index=None):
    if industry_name is None:
        merged = random_stocks(merged)
    merged = random_stocks(merged, industry_name, index)
    hr = hr.reindex(merged['CompanyCode'])
    factor_loading = factor_loading.reindex(merged['CompanyCode'])
    industry = industry.reindex(merged['CompanyCode'])
    return merged, hr, factor_loading, industry


def main(filename, s=False):
    # pre processing
    data = pd.read_csv('expo_test.csv')
    l = ['ST保千里', '万科Ａ', '京东方Ａ', '华侨城Ａ', '海虹控股']
    l1 = ['*ST保千', '万科A', '京东方A', '华侨城A', '海航控股']
    d = {l[i]: l1[i] for i in range(5)}

    def f(x):
        x = x.replace(' ', '')
        return d[x] if x in l else x

    data['TickerName'] = data['TickerName'].apply(f)
    months = data['Month'].unique()
    factors_t = data[data['Month'] == months[-1]]

    data2 = pd.read_csv('hs300.csv', encoding='GBK')
    data2['简称'] = data2['简称'].apply(lambda x:x.replace(' ', ''))
    # merge the two DataFrames
    merged = pd.merge(factors_t, data2, left_on='TickerName', right_on='简称', how='inner')

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

    # get risk structure
    B = np.asarray(merged['权重（%）↓'])/100
    op.set_base(B)
    op.set_risk(0.01)
    op.set_up(0.07)
    op.set_deviate(0.01)
    # get weight
    w1 = pd.Series(op.optimize_returns() * 100, index=merged['简称'])
    op.set_risk(0.03)
    w2 = pd.Series(op.optimize_returns() * 100, index=merged['简称'])
    op.set_risk(0.05)
    w3 = pd.Series(op.optimize_returns() * 100, index=merged['简称'])
    op.set_risk(0.07)
    w4 = pd.Series(op.optimize_returns() * 100, index=merged['简称'])
    weight = pd.concat([w1, w2, w3, w4], axis=1)
    if not s:
        weight = weight.reindex(data2['简称']).fillna(value=0)  # when sample no need
    data2.index = data2['简称']
    weight = weight.apply(lambda x: x + data2['权重（%）↓']).dropna()

    # produce result
    result = pd.merge(weight, data2, right_index=True, left_index=True, how='inner')
    result.to_csv(filename, encoding='GBK', float_format='%.2f')


def test():
    data = pd.read_csv('expo_test.csv')
    factors = list(data.columns.drop(['Ticker', 'CompanyCode', 'TickerName', 'SecuCode', 'IndustryName',
                                      'CategoryName', 'Date', 'Month', 'Return', 'PCF']))

    # remove redundant variables
    factors.remove('AnalystROEAdj')
    factors.remove('FreeCashFlow')

    op = Optimus(data, factors)

    op.set_names(freq='Month', returns='Return', company='TickerName', industry='IndustryName')

    op.factor_model()
    # op.print_private()
    # print(op.get_components())
    # print(op.get_industry_dummy())
    print(op.max_returns(0.7, up=0.01))
    b = np.ones(288) / 288
    industry = op.get_industry_dummy()
    print(op.max_returns(0.07, b, 0.01, industry, 0.01))
    print(op.min_risk(0.1, up=0.01))
    print(op.min_risk(0.1, b, 0.01, industry, 0.01))


if __name__ == '__main__':
    test()
