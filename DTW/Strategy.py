import numpy as np


OPT = {
    'data_dir': 'data/中信一级行业指数/zz800.csv',           # 收盘价数据
    'nav_sdir': 'output/zz800_s1.csv',                      # 净值结果存储位置
    'w_sdir': 'output/zz800_s1_w.csv',                        # 权重结果存储位置
    'base_dir': 'data/3145.csv',                            # 基准数据位置
    'begin': 201008,                                        # 回测开始月份
    'end': 201808,                                          # 回测结束月份
    'return_win': 1,                                        # 预测收益窗口大小
    'trend_win': 6,                                         # 趋势窗口
    'method': 'filter',                                     # 计算方法，不需要改动
    'pos': np.ones(50) / 50                                 # 等权无行业中性时的权重配置
}
