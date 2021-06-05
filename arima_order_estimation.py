import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pylab as plt
from pmdarima import auto_arima
from time import perf_counter
import pmdarima as pm

from lib import get_data



data = get_data('2021-02-02T00:00', '2021-02-04T00:00', ['eth'])


x = data['timestamp'].values
df = data['eth_close']
print(df)


model = auto_arima(df,
                   start_P=0,
                   start_q=0,
                   max_p=3,
                   max_q=3,
                   seasonal=False,
                   d=1,
                   trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)

model.summary()
#params = model.get_params()


t0 = perf_counter()
forecast_model = ARIMA(df[:-80],
                    order=(1,0,1),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
t1 = perf_counter()
print(t1-t0)

model_fit = forecast_model.fit()
t2 = perf_counter()
print(t2-t1)

'''
arima = pm.ARIMA(order=params['order'])
t3 = perf_counter()
print(t3-t2)



arima.fit(df[:-200])
t4 = perf_counter()
print(t4-t3)

model.predict(200)
t5 = perf_counter()
print(t5-t4)
'''



yf = model_fit.get_forecast(80, typ='levels')
ci = yf.conf_int()

ax = df[:-80].plot(label='observed', figsize=(20, 15))
yf.predicted_mean.plot(ax=ax, label='Forecast')
data['eth_close'][:].plot(ax=ax, label='actual')
ax.fill_between(ci.index,
                ci.iloc[:, 0],
                ci.iloc[:, 1], color='k', alpha=.25)


plt.legend()
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
sarimax_model.fit(method='bfgs', disp=False,    )
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
'''

