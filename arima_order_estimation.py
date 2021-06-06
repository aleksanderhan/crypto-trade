import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pylab as plt
from pmdarima import auto_arima
from time import perf_counter
import pmdarima as pm

from lib import get_data



data = get_data('2021-02-02T00:00', '2021-02-03T00:00', ['eth'])
print()

x = data['timestamp'].values
df = data['eth_close']
print(df)

lookback_interval = 3


print(df.loc[0:lookback_interval])
print(df.at[lookback_interval])


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

forecast, conf_int = model.predict(n_periods=10, return_conf_int=True)


print(forecast)
print(conf_int)

ci = pd.DataFrame(conf_int)


ax = df[:-80].plot(label='observed', figsize=(20, 15))
pd.DataFrame(forecast).plot(ax=ax, label='Forecast')
data['eth_close'][:].plot(ax=ax, label='actual')
ax.fill_between(ci.index,
                ci.iloc[:, 0],
                ci.iloc[:, 1], color='k', alpha=.25)


plt.legend()
plt.show()





