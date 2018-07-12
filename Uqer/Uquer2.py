#**********************************************#
# 双均线突破策略 - 无止盈止损
# 短周期 MA 上穿长周期 MA，形成做多信号，买入开仓；
# 短周期 MA 下穿长周期 MA，形成做空信号，卖出开仓；
#**********************************************#
import talib
import pandas as pd
import numpy as np
import quartz_futures as qf
from quartz_futures.api import *

### 参数初始化
universe = ['RBM0']               # 策略证券池
start = '2013-07-01'
end   = '2017-06-30'
refresh_rate = 1                 # 调仓周期
freq = 'd'                          # 调仓频率：s -> 秒；m-> 分钟；d-> 日；

## 自动生成保证金比例： margin_rate
margin_ratio = DataAPI.FutuGet(ticker = universe, field = ['ticker','tradeMarginRatio'], pandas = '1')
margin_rate = dict(zip(margin_ratio.ticker.tolist(), [0.01*index for index in margin_ratio.tradeMarginRatio.tolist()]))

accounts = {
    'futures_account': AccountConfig(account_type='futures', capital_base=10000, margin_rate=margin_rate)
}

### 策略初始化函数，一般用于设置计数器，回测辅助变量等。
def initialize(context):
    pass

### 回测调仓逻辑，每个调仓周期运行一次，可在此函数内实现信号生产，生成调仓指令。
def handle_data(context):
    futures_account = context.get_account('futures_account')

    if main_contract_mapping_changed(context, futures_account):
        return

    symbol = context.get_symbol('RBM0')
    amount = 1

    current_long = futures_account.get_positions().get(symbol, dict()).get('long_amount', 0)
    current_short= futures_account.get_positions().get(symbol, dict()).get('short_amount', 0)

    history_data = context.history(symbol=symbol, attribute=['closePrice', 'openPrice', 'lowPrice', 'highPrice'], time_range=30, freq='1d')[symbol]

    MA_S = talib.MA(history_data['closePrice'].apply(float).values, timeperiod = 5)
    MA_L = talib.MA(history_data['closePrice'].apply(float).values, timeperiod = 10)
    if MA_S[-1] > MA_L[-1] and MA_S[-2] < MA_L[-2]:
        if current_short > 0:
            futures_account.order(symbol, current_short, 'close')
        if current_long < amount:
            futures_account.order(symbol, amount, 'open')

    if MA_S[-1] < MA_L[-1] and MA_S[-2] > MA_L[-2]:
        if current_long > 0:
            futures_account.order(symbol, -current_long, 'close')
        if current_short < amount:
            futures_account.order(symbol, -amount, 'open')


def main_contract_mapping_changed(context, futures_account):
    '''
        处理移仓换月的情况
    '''
    if context.mapping_changed('RBM0'):
        symbol_before, symbol_after = context.get_rolling_tuple('RBM0')
        if futures_account.get_position(symbol_before):
            futures_account.switch_position(symbol_before, symbol_after)
            return True

    return False
