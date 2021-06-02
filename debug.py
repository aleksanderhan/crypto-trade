import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pylab as plt
from pmdarima import auto_arima
from time import perf_counter

from lib import get_data

'''
x = np.linspace(-2*np.pi, 2*np.pi, 100)

y = []
for i in x:
	y.append(np.sin(i) + i)

y = df = pd.DataFrame(y)

forecast_model = SARIMAX(y,order=(1, 0, 1),
              seasonal_order=(2, 1, 0, 12),
              enforce_stationarity=False,
              enforce_invertibility=False)
model_fit = forecast_model.fit(method='bfgs', disp=False)

#yf = model_fit.get_forecast(100)

yp = model_fit.predict(start=101, end=200, typ='levels')
print(len(yp))

df.plot(figsize=(12,8),legend=True)
yp.plot(legend=True)


#plt.plot(x, yf.mean)

plt.show()

'''

df = np.array([-0.00021112, -0.0005432 ])


forecast_model = SARIMAX(df,order=(1, 0, 1),
              seasonal_order=(2, 1, 0, 12),
              enforce_stationarity=False,
              enforce_invertibility=False)


t0 = perf_counter()
model_fit = forecast_model.fit(method='bfgs', disp=False, start_params=[0, 0, 0, 1, 1])
t1 = perf_counter()
model2 =forecast_model.fit(method='lbfgs', disp=False, start_params=[0, 0, 0, 1, 1])
t2 = perf_counter()
model2 =forecast_model.fit(method='lbfgs', disp=False, start_params=[0, 0, 0, 1, 1], simple_differencing = True)
t3 = perf_counter()


print(t1-t0)
print(t2-t1)
print(t3-t2)
