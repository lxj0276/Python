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


def main():
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

    data2 = pd.read_excel('weight.xlsx')
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

    # get history factor returns
    hfr = op.hist_factor_returns()
    hr = op.hist_residuals(hfr)

    # get factor loadings at period T
    factor_loading = op.get_factor_loading_t()

    # predict factor returns at period T+1
    pfr = op.predict_factor_returns(hfr, 'hpfilter')

    industry = op.get_industry_dummy()

    # sample
    # merged, hr, factor_loading, industry = sample(merged, hr, factor_loading, industry)

    # predict stock returns
    psr = op.predict_stock_returns(factor_loading, pfr)

    # get risk structure
    B = np.asarray(merged['权重（%）↓'])/100
    rs = op.risk_structure(hfr, hr, factor_loading)

    # get weight
    w1 = pd.Series(op.max_returns(psr, rs, 0.01, B, 0.07, industry, 0.01) * 100, index=merged['简称'])
    w2 = pd.Series(op.max_returns(psr, rs, 0.03, B, 0.07, industry, 0.01) * 100, index=merged['简称'])
    w3 = pd.Series(op.max_returns(psr, rs, 0.05, B, 0.07, industry, 0.01) * 100, index=merged['简称'])
    w4 = pd.Series(op.max_returns(psr, rs, 0.07, B, 0.07, industry, 0.01) * 100, index=merged['简称'])
    weight = pd.concat([w1, w2, w3, w4], axis=1)
    weight = weight.reindex(data2['简称']).fillna(value=0)
    data2.index = data2['简称']
    weight = weight.apply(lambda x: x + data2['权重（%）↓']).dropna()

    # produce result
    result = pd.merge(weight, data2, right_index=True, left_index=True, how='inner')
    result.to_csv('TEST.csv', encoding='GBK', float_format='%.2f')


def main2():
    s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
    data2 = pd.read_excel('weight.xlsx')
    count = data2.groupby(data2['Wind行业']).apply(lambda x: len(x))
    print(count)
    print(random_stocks(data2, 'Wind行业'))


if __name__ == '__main__':
    main()

