from WindPy import *
import pandas as pd
import numpy as np

import TriPeriods as Tri


def download():
    sw_indus = pd.read_csv('data/sw_indus.csv', encoding='GBK')

    # 如果返回 .ErrorCode = -103，以管理员身份运行此程序及脚本
    w.start()

    begin = '2000-01-01'
    last_daily_date = '2018-02-07'
    last_month_date = '2018-01-31'

    daily_data = w.wsd(list(sw_indus['indus_code']), 'close', begin, last_daily_date, 'Period=D')
    monthly_dates = w.tdays(begin, last_month_date, 'Period=M')

    daily_dates = np.asarray([Tri.date_num(i.strftime('%Y-%m-%d')) for i in daily_data.Times])
    monthly_dates = np.asarray([Tri.date_num(i.strftime('%Y-%m-%d')) for i in monthly_dates.Times])

    np.savetxt('mydata/close.csv', np.asmatrix(daily_data.Data).T, delimiter=',')
    np.savetxt('mydata/daily_dates.csv', daily_dates, delimiter=',')
    np.savetxt('mydata/monthly_dates.csv', monthly_dates, delimiter=',')


if __name__ == '__main__':
    # download()
    mat = pd.DataFrame(np.empty((3, 3)), dtype='int', index=['a', 'b', 'c'])
    mat.iloc[:, 1] = [1, 2, 3]
    print(mat)

