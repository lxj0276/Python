import pandas as pd
import numpy as np
import statsmodels.api as sm
from cvxopt import matrix
from Optimus import Optimus


def main():
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
    pd.set_option('max_rows', 288)
    data2 = pd.read_excel('weight.xlsx')
    data2['简称'] = data2['简称'].apply(lambda x:x.replace(' ', ''))
    merged = pd.merge(factors_t, data2, left_on='TickerName', right_on='简称', how='inner')
    print(len(merged))
    print(sum(merged['权重（%）↓']))

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
    factor_loading = op.get_factor_loading_t()
    print(factor_loading)

    # predict factor returns at period T+1
    pfr = op.predict_factor_returns(hfr, 'hpfilter')
    print(pfr)

    # predict stock returns
    psr = op.predict_stock_returns(factor_loading, pfr)
    print(psr)

    # get risk structure
    B = np.asarray(merged['权重（%）↓'])/100
    rs = op.risk_structure(hfr, factor_loading)
    print(rs)

    industry = op.get_industry_dummy()
    weight = pd.Series(op.max_returns(psr, rs, 0.14, B, 0.07, industry, 0.01) * 100, index=merged['简称'])
    weight = weight.reindex(data2['简称']).fillna(value=0)
    data2.reindex(data2['简称'])
    data2['Weight'] =np.asarray(data2['权重（%）↓']) + np.asarray(weight)
    data2.to_csv('out.csv', encoding='GBK')


if __name__ == '__main__':
    g = (i for i in [1, 2, 3])

    def f(g):
        return next(g)**2
    s = None

    s = pd.Series([f(g) for i in range(3)])

    print(s)
