# -*- coding:utf-8 -*-

from WindPy import *

import pandas as pd
import numpy as np


def get_data():
    codes = list(pd.read_csv('data/fund_codes.csv')['Code'])
    orders = [1, 2, 3]

    w.start()

    managers = [w.wss(codes, 'fund_manager_longestfundmanager', order=i) for i in orders]
    start_dates = [w.wss(codes, 'fund_manager_startdate', order=i) for i in orders]

    columns = ['m1', 'm2', 'm3', 'd1', 'd2', 'd3']
    manager_data = pd.DataFrame(index=codes, columns=columns)
    for i in range(3):
        manager_data.iloc[:, i] = np.asarray(managers[i].Data[0])
        manager_data.iloc[:, i+3] = np.asarray(start_dates[i].Data[0])

    manager_data = manager_data.applymap(lambda x: x if x is not None else np.nan)

    manager_data.loc[manager_data['m2'].isnull(), 'd2'] = np.nan
    manager_data.loc[manager_data['m3'].isnull(), 'd3'] = np.nan

    name = '安昀'
    m1 = manager_data[manager_data['m1'] == name][['m1', 'd1']].rename(columns={'m1': 'm', 'd1': 'd'})
    m2 = manager_data[manager_data['m2'] == name][['m2', 'd2']].rename(columns={'m2': 'm', 'd2': 'd'})
    m3 = manager_data[manager_data['m3'] == name][['m3', 'd3']].rename(columns={'m3': 'm', 'd3': 'd'})
    manager = pd.concat([m1, m2, m3])
    print(manager)

    begin = manager['d'].min()
    fund = manager['d'].idxmin()
    print(begin, fund)
    end = '2018-08-01'
    index = ['000828.SH', '000829.CSI', '399666.SZ', '399665.SZ']
    index_data = w.wsd(index, 'close', begin, end)
    index_data = pd.DataFrame(np.asmatrix(index_data.Data).T, columns=index, index=index_data.Times)
    index_data = (index_data / index_data.shift(1) - 1).dropna()

    def func(x):
        series = pd.Series(0, index=x.index)
        series[x.idxmax()] = 1
        return series
    index_data = index_data.apply(func, axis=1)

    fund_data = w.wsd(codes, 'close', begin, end)
    fund_data = pd.DataFrame(np.asmatrix(fund_data.Data).T, columns=codes, index=fund_data.Times)

    total = fund_data.count(axis=1)
    rank = (fund_data > fund_data[fund]).apply(lambda x: x.any(), axis=1)
    print(rank.any())
    w.stop()


if __name__ == '__main__':
    get_data()
    # writer = pd.ExcelWriter('output/test.xlsx')
    # df = pd.DataFrame(np.arange(9).reshape(3, 3))
    # print(df)
    # df.to_excel(writer, 'Sheet2')
    # writer.save()
