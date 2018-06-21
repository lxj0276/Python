import pandas as pd
import LargeAsset as La
from Optimus import Optimus


# 大类资产配置
data = pd.read_csv('data/IndexPrice.csv', index_col='DATES')
data.index = pd.to_datetime(data.index.map(str))
data['PCTCHG'] /= 100

# 处理数据
res1 = La.predict_month_return(data)
res2 = La.get_risk(res1)['predict']

predict_returns = res1.loc['2015-03']['predict']
predict_risk = res2[201503]

# 建立模型
model = Optimus()
model.psr = predict_returns
model.rs = predict_risk
print(model.min_risk(0.01))