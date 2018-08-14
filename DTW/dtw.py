import numpy as np
import pandas as pd
import os

from time import time
from config import R

os.environ['R_USER'] = R['R_USER']
os.environ['R_HOME'] = R['R_HOME']


class RConsole:
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr
    rpy2.robjects.numpy2ri.activate()

    # Set up R namespaces
    R = rpy2.robjects.r
    importr('dtw')

    def __init__(self):
        pass

    def dtw(self, query, template):
        if type(query) != np.ndarray:
            query = np.asarray(query)
        if type(template) != np.ndarray:
            template = np.asarray(template)

        # Calculate the alignment vector and corresponding distance
        alignment = self.R.dtw(query, template, keep=True)
        dist = alignment.rx('distance')[0][0]
        return dist


def cal_similarity(arr1, arr2, freq='M', num=3):
    try:
        if type(arr1.index) != pd.DatetimeIndex:
            arr1.index = str(arr1.index)
            arr1.index = pd.to_datetime(arr1.index)
        if type(arr2.index) != pd.DatetimeIndex:
            arr2.index = str(arr2.index)
            arr2.index = pd.to_datetime(arr2.index)
    except ValueError:
        print('计算失败，输入序列的索引非日期索引')
        return None

    arr2 = arr2.reindex(arr1.index)

    if freq == 'M':
        cut = arr1.index.year * 100 + arr2.index.month
    elif freq == 'Y':
        cut = arr1.index.year
    else:
        cut = arr1.index

    assert len(cut.unique()) > num, '输入的序列不足 {} 年/月/日'.format(num)

    cut_point = cut.unique()[-num]
    cut = cut >= cut_point

    r = RConsole()
    arr1, arr2 = arr1[cut], arr2[cut]
    arr1 = arr1 / arr1[0]
    arr2 = arr2 / arr2[0]

    arr1.plot()
    arr2.plot()

    return r.dtw(arr1, arr2)


def get_data():
    # pre-processing the data
    index_list = [3145, 3159, 4978, 6455, 14599]
    data_list = [pd.read_csv('data/{}.csv'.format(i)) for i in index_list]
    data = pd.concat([i['close'] for i in data_list], axis=1)
    data.index = pd.to_datetime(data_list[0]['Date'])
    data.columns = index_list
    data.dropna(inplace=True)
    data = data / data.iloc[0, :]
    return data


if __name__ == '__main__':
    data = get_data()

    import matplotlib.pyplot as plt

    print(cal_similarity(data[3145], data[3159], 'M', 3))
    print(cal_similarity(data[3145], data[4978], 'M', 3))

    plt.show()