import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pd_check():
    result = ts.get_hist_data('600000')
    t = type(result)
    columns = result.columns

    # 生成一列年份数据
    result['year'] = result.index
    result['year'] = result['year'].apply(lambda obj:obj[:4])  # 应用 apply 函数
    result['date'] = result.index
    data = result[result['date'].apply(lambda obj:obj[:4]=='2018')]  # 索引筛选

    # 排序和排名
    result.sort_index(by='open')  # 按照某个索引排序
    result['open'].rank(method='first')  # 用来破坏平级关系，否则用 平均排名 表示

    # 描述性统计
    result.mean() # 默认是列方向
    result['open'].idxmax()  # 查看最高值的日期、索引
    result['open'].idxmin()  # 查看最低值的日期、索引
    result['close'].quantile(0.5)  # 分位数
    result['p_change'].cumprod()  # 累计积
    result['close'].diff()  # 一阶查分
    result['close'].pct_change()  # 百分比变化
    result['close'].corr(result['open'])  # 相关系数, pd.DataFrame.corrwith()
    result['close'].cov(result['open'])  # 协方差

    # 处理缺失数据
    result.dropna()
    result = result[result.notnull()]
    result.fillna(0, method='ffill', inplace=True)  # inplace 就地修改

    # 读取
    pd.read_csv()
    pd.read_table()
    '''
    sep/deliminator:regex
    encoding
    '''

    # 合并数据集
    pd.merge(result, data, on='date', how='outer')
    '''
    right_index/left_index : True/False
    on/left_on/right_on : [...]
    how : outer, inner, left, right
    '''
    result.join(data)  # 默认取索引的版本
    pd.concat([data, result], axis=1)  # 轴向连接, axis=1表示行方向上进行
    '''
    axis : 0/1 
    join_axes : ['a']
    '''
    np.where(result.isnull(), result, data)
    result.combine_first(data)
    '''
    others:
    stack/unstack
    drop_duplicates
    duplicates : True/False Series
    '''
    result['close'].map(str.lower)  # 类似于apply
    result['close'].replace()

    # 字符串(支持正则)
    data['data'].str.contains('mail')
    pattern = '.*'
    data['data'].str.findall(pattern)
    matches = data['data'].str.match(pattern)  # 匹配
    matches.str.get(1)
    matches = matches.str[0]  # 取出匹配结果

    # 分组聚合
    grouped = result.groupby(result['year'])  # 分组
    grouped.mean()  # 聚合

    result['month'] = result.index.map(lambda x: x[5:7])
    means = result.groupby([result['year'], result['month']]).mean()  # 两个分组，生成层次化索引
    means.unstack()  # 纵向生成横向 DataFrame

    grouped = result.groupby([result['year'], result['month']])
    for (k1, k2), group in grouped:
        # 分组迭代
        group.size()

    grouped.transform(lambda x:x-x.mean())

    factor = pd.cut(result.open, 4)  # 分位数和桶分析
    result.groupby(factor).apply(lambda x:x.min())


def ts_check():
    pass


def main():
    pd_check()


if __name__ == '__main__':
    main()
