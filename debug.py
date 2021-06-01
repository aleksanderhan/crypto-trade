import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pylab as plt







def _get_forecast(df):
    past_close_values = df


    forecast_model = SARIMAX(np.nan_to_num(np.diff(np.log(past_close_values))))
    model_fit = forecast_model.fit(method='bfgs', disp=False)
    forecast = model_fit.get_forecast(steps=10, alpha=(1 - 0.8))

    return forecast




df = [34557.95, 34525.95, 3]




f = _get_forecast(df)