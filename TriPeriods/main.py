import pandas as pd
import TriPeriods as Tri


if __name__ == '__main__':
    sw_indus = pd.read_csv('data/sw_indus.csv', encoding='GBK')
    close = pd.read_csv('data/close.csv', header=None)
    daily_dates = pd.read_csv('data/daily_dates.csv', header=None)
    monthly_dates = pd.read_csv('data/monthly_dates.csv', header=None)
    indus_num = 28
    data = {
        'close': close,
        'indus_num': indus_num,
        'daily_dates': daily_dates,
        'monthly_dates': monthly_dates
    }

    Tri.gen_global_param(data, sw_indus)

    print(Tri.GlobalParam.daily_close)
    print(Tri.GlobalParam.daily_dates)
    print(Tri.GlobalParam.base_nav)
    print(Tri.GlobalParam.monthly_indus)
    print(Tri.GlobalParam.monthly_base)
    print(Tri.GlobalParam.monthly_dates)
    print(Tri.GlobalParam.yoy_indus)
    print(Tri.GlobalParam.yoy_base)
    print(Tri.GlobalParam.yoy_dates)

    Tri.bktest_unit()
    print(Tri.BktestParam.predict_dates)
    print(Tri.BktestParam.start_date)
    print(Tri.BktestParam.end_date)
    print(Tri.BktestParam.refresh_dates)
    print(Tri.BktestParam.start_loc)
    print(Tri.BktestParam.end_loc)
    print(Tri.BktestParam.back_dates)
    print(Tri.BktestParam.monthly_refresh_dates)








