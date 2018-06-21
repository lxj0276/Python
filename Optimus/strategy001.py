import pandas as pd
from Optimus import Optimus


# 读取数据
data = pd.read_csv('data/pool.csv')
data.dropna(inplace=True)
data.drop(['Index_y', 'Index_x'], axis=1, inplace=True)
factors = list(data.columns.drop(
    ['Ticker', 'CompanyCode', 'TickerName', 'SecuCode', 'IndustryName', 'CategoryName', 'Date', 'Month', 'Return',
     'PCF']))

factors.remove('AnalystROEAdj')
factors.remove('FreeCashFlow')

# 多因子过程
model = Optimus(data, factors)
model.set_names(freq='Month')

# 以下过程除优化外都封装在了函数 model.factor_model()中

model.hfr = model.hist_factor_returns()
model.hr = model.hist_residuals(model.hfr)

# 获取当期因子暴露
model.factor_loading = model.get_factor_loading_t()

# 预测当期的因子收益
model.pfr = model.predict_factor_returns(model.hfr, 'hpfilter')

# 风险结构
model.psr = model.predict_stock_returns(model.factor_loading, model.pfr)
model.rs = model.risk_structure(model.hfr, model.hr, model.factor_loading)

# 优化
print(model.min_risk(0.01))


