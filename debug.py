import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pylab as plt
from pmdarima import auto_arima
from time import perf_counter

from lib import get_data


x = np.linspace(-2*np.pi, 2*np.pi, 100)

y = []
for i in x:
	y.append(np.sin(i) + i)

df = pd.DataFrame(y)
'''
forecast_model = SARIMAX(df,order=(3, 1, 3),
              seasonal_order=(3, 1, 3, 3),
              enforce_stationarity=False,
              enforce_invertibility=False)

forecast_model = ARIMA(df,order=(3, 1, 3),
              enforce_stationarity=False,
              enforce_invertibility=False)



model_fit = forecast_model.fit()

yf = model_fit.get_forecast(100, typ='levels')

yp = model_fit.predict(start=101, end=200, typ='levels')
print(len(yp))

df.plot(figsize=(12,8),legend=True)
yp.plot(legend=True)
yf.predicted_mean.plot(legend=True)


#plt.plot(x, yf.mean)

plt.show()

'''

df = np.array([-0.00021112, -0.0005432 ])


sarimax_model = SARIMAX(df,order=(1, 0, 1),
              seasonal_order=(2, 1, 0, 12),
              enforce_stationarity=False,
              enforce_invertibility=False)

arima_model = ARIMA(df,order=(1, 0, 1),
              enforce_stationarity=False,
              enforce_invertibility=False)

t0 = perf_counter()
sarimax_model.fit(method='bfgs', disp=False,	)
t1 = perf_counter()
sarimax_model.fit(method='lbfgs', disp=False, start_params=[0, 0, 0, 1, 1])
t2 = perf_counter()
sarimax_model.fit(method='lbfgs', disp=False, start_params=[0, 0, 0, 1, 1], simple_differencing = True)
t3 = perf_counter()
arima_model.fit(start_params=[0, 0, 0, 1, 1])
t4 = perf_counter()

print(t1-t0)
print(t2-t1)
print(t3-t2)
print(t4-t3)