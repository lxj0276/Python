start = '2017-01-01'                       # 回测起始时间
end = '2018-01-01'                         # 回测结束时间
universe = ['600036.XSHG', '601166.XSHG']  # 证券池
benchmark = 'HS300'                        # 策略参考标准
freq = 'd'                                 # 每日调仓
refresh_rate = 1                           # 调仓频率

# 配置账户信息
accounts = {
    'my_account': AccountConfig(account_type='security',
                                capital_base=1000000,
                                commission = Commission(buycost=0.001, sellcost=0.002, unit='perValue'),
                                slippage = Slippage(value=0.001, unit='perValue'))
}

# 获取PB值对应的仓位
def cal_position(PB):
    if PB < 1.0:
        return 0.5
    if PB > 2.0:
        return 0
    else:
        return (PB - 1.0) if (PB - 1.0) < 0.5 else 0.5

def initialize(context):
    pass

# 回测
def handle_data(context):
    my_account = context.get_account('my_account')
    data = context.history(['601166.XSHG', '600036.XSHG'], ['PB'], 1, freq='1d', rtype='array', \
                           style='sat')
    pb0 = data['601166.XSHG']['PB'][0]
    pb1 = data['600036.XSHG']['PB'][0]

    # 根据目标持仓权重，逐一委托下单
    my_account.order_pct_to('601166.XSHG', cal_position(pb0))
    my_account.order_pct_to('600036.XSHG', cal_position(pb1))
    
