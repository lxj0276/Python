from Optimus import Optimus
import pandas as pd
import numpy as np
from itertools import product

date = ['2018-01-01', '2018-01-02', '2018-01-03']
company = ['A', 'B', 'C']
data = pd.DataFrame(list(product(date, company)), columns=['date', 'company'])
data[['PE', 'PB', 'PS']] = pd.DataFrame(np.arange(27).reshape(9, 3))
factors = ['PE', 'PB', 'PS']
data['return'] = np.arange(9)
op = Optimus(data, factors)
op.rank_factors()
print(op.x)
op.set_names(freq='date', company='company', returns='return')
hfr = op.hist_factor_returns()
print("hfr:")
print(hfr)
pfr = op.predict_factor_returns(hfr, method='ewma', arg=0.5)
print("pfr:")
print(pfr)
hr = op.hist_residuals(hfr)
print("hr:")
print(hr)
pfl = op.predict_factor_loadings(method="ewma", arg=0.5)
print("pfl:")
print(pfl)
psr = op.predict_stock_returns(pfl, pfr)
print("psr:")
print(psr)
rs = op.risk_structure(hist_factor_returns=hfr, hist_residuals=hr, factor_loadings=pfl)
print("rs:")
print(rs)
B = np.ones(3) / 3
sol = op.max_returns(psr, rs, 0.01, B, 0.5)
print("sol:")
print(sol)