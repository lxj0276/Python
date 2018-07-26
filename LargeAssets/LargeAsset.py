import pandas as pd
import numpy as np


def __predict(freq):
    def decorate(func):
        def predict_at_freq(data):
            func()
            grouped = data.groupby(data['TICKER'])
            index = pd.Series(np.nan, index=data.index.drop_duplicates()).asfreq(freq)

            real = pd.DataFrame(columns=data['TICKER'].unique(), index=index.index)
            for ticker, group in grouped:
                real[ticker] = (group['PCTCHG'] + 1).resample(freq).prod()

            assert len(real) > (12 if freq == 'M' else 0), 'Data Too Short'

            predicts = pd.DataFrame(columns=data['TICKER'].unique(), index=index.index)
            for i in range((1 if freq == 'M' else 1), len(real) + 1):
                predicts.iloc[i - 1] = (real.iloc[(i - 12 if freq == 'M' and i > 12 else 0): i]
                                        .ewm(alpha=0.8).mean().iloc[-1])

            predicts = predicts.shift(1)
            res = pd.concat([real.stack(), predicts.stack()], axis=1)
            res.columns = ['actual', 'predict']
            res['predict'] = res['predict'].astype('float64')
            res[['actual', 'predict']] -= 1
            res.index.names = ['date', 'ticker']
            res = pd.concat([res, res.index.to_frame()], axis=1)

            def f(x):
                if freq == 'M':
                    return x.year * 100 + x.month
                else:
                    return x.year
            res['period'] = res['date'].apply(f)
            return res

        return predict_at_freq
    return decorate


@__predict('M')
def predict_month_return():
    pass


@__predict('Y')
def predict_year_return():
    pass


def get_risk(returns):
    data = returns.unstack()
    dates = returns['period'].unique()

    def na(d):
        if d.iloc[:, -1].notna().any() and d.iloc[:, -1].isna().any():
            d.iloc[:, -1] = d.iloc[:, -1].fillna(d.iloc[:, -1].mean(skipna=True))
        else:
            d = d.dropna(axis=1)
        return d

    actual, predict = {}, {}
    for i in range(2, len(data['actual'])+1):
        dt_real = data['actual'][i-12 if i > 12 else 0: i]
        dt_predict = data['predict'][i-12 if i > 12 else 0: i]
        dt_predict = pd.concat([dt_real.iloc[:-1], dt_predict.iloc[-1:]])

        actual[dates[i - 1]] = np.cov(na(dt_real).T)
        predict[dates[i - 1]] = np.cov(na(dt_predict).T)

    return {'actual': actual, 'predict': predict}


if __name__ == '__main__':
    data = pd.read_csv('data/IndexPrice.csv', index_col='DATES')
    data.index = pd.to_datetime(data.index.map(str))
    data['PCTCHG'] /= 100
    res = predict_year_return(data)
    print(res)
    print(get_risk(res))

