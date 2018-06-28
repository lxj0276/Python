import pandas as pd
import TriPeriods as Tri


if __name__ == '__main__':
    sw_indus = pd.read_csv('data/sw_indus.csv', encoding='GBK')
    close = pd.read_csv('mydata/close.csv', header=None)
    daily_dates = pd.read_csv('mydata/daily_dates.csv', header=None)
    monthly_dates = pd.read_csv('mydata/monthly_dates.csv', header=None)
    indus_num = len(sw_indus)
    data = {
        'close': close,
        'indus_num': indus_num,
        'daily_dates': daily_dates,
        'monthly_dates': monthly_dates
    }

    Tri.gen_global_param(data, sw_indus)

    Tri.bktest_unit()

    Tri.portfolio()





