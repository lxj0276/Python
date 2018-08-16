import numpy as np


OPT = {
    'data_dir': 'data/中信一级行业指数/hs300.csv',
    'nav_sdir': 'output/hs300_nav_6150.csv',
    'w_sdir': 'output/hs300_w_6150.csv',
    'begin': 201001,
    'end': 201808,
    'return_win': 1,
    'trend_win': 6,
    'method': 'filter',
    'pos': np.ones(50) / 50
}