import pandas as pd

from back import Context
from model import Model
from strategy import OPT

from time import time


# pd.set_option('display.max_rows', 3000)


def signal(context, date, refresh, industry):
    if date in refresh:
        print("{}调仓".format(date))
        data = context.Add['data']

        num = industry.groupby('Industry').count()
        ind_num = (num / 5).applymap(round)
        ind_num[ind_num['Code'] == 0] = 1
        total = ind_num.sum()['Code']

        begin = time()

        model = Model(data[:date], OPT['trend_win'], OPT['return_win'])
        result = model.train(OPT['method'])
        ranks, _ = result.filter(300)

        df = pd.DataFrame(columns=['industry', 'rank', 'weight'])
        df['rank'] = ranks
        df['industry'] = industry['Industry']
        for i in ind_num.index:
            stocks = df.index[df['industry'] == i][:ind_num.loc[i, 'Code']]
            df.loc[stocks, 'weight'] = 1 / total

        print("耗时{:.2f}秒".format(time()-begin))
        weight = df['weight']
        weight = weight.reindex(data.columns).fillna(0)
        return True, weight
    else:
        return False, None


def main():
    data = pd.read_csv(OPT['data_dir'], engine='python', index_col=0)
    industry = pd.read_excel('data/中信一级行业指数/industry.xlsx', sheet_name=0)   # 行业数据
    industry.index = industry['Code']
    date_index = pd.to_datetime(data.index)

    # 生成月度的索引序列
    month = date_index.year * 100 + date_index.month
    starts = date_index[month.to_series().diff() != 0]  # 每月月初
    cut = month.unique() >= OPT['begin']
    cut[month.unique() >= OPT['end']] = False
    starts = [i.strftime('%Y-%m-%d') for i in starts[cut]]

    context = Context()
    context.Add['data'] = data
    context.Add['signal_params'] = {'refresh': starts,
                                    'industry': industry}
    context.BktestParam['signal'] = signal
    context.GlobalParam['daily_close'] = data.loc[starts[0]: starts[-1]]

    begin = time()

    w = context.cal_long_weight()
    nav = context.cal_nav()

    print("total time : {:.2f}".format(time()-begin))

    w.to_csv(OPT['w_sdir'])
    nav.to_csv(OPT['nav_sdir'])


if __name__ == '__main__':
    main()
