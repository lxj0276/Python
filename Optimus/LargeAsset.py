import pandas as pd
import numpy as np
from Optimus import Optimus


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
            for i in range((12 if freq == 'M' else 1), len(real) + 1):
                predicts.iloc[i - 1] = real.iloc[(i - 12 if freq == 'M' else 0): i].ewm(alpha=0.8).mean().iloc[-1]

            predicts = predicts.shift(1)
            res = pd.concat([real.stack(), predicts.stack()], axis=1)
            res.columns = ['actual', 'predict']
            res[['actual', 'predict']] -= 1
            res.index.names = ['date', 'ticker']
            res = pd.concat([res, res.index.to_frame()], axis=1)

            def f(x):
                if freq == 'M':
                    return x.year * 100 + x.month
                else:
                    return x.year
            res[freq] = res['date'].apply(f)
            return res

        return predict_at_freq
    return decorate


@__predict('M')
def predict_month_return():
    pass


@__predict('Y')
def predict_year_return():
    pass


def predict_risk(returns, freq):
    pass


def main():
    data = pd.read_csv('data/IndexPrice.csv', index_col='DATES')
    data.index = pd.to_datetime(data.index.map(str))
    data['PCTCHG'] /= 100
    res = predict_month_return(data)
    print(res.swaplevel(0, 1).loc[3145])


if __name__ == '__main__':
    main()
