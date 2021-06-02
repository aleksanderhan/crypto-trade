import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pylab as plt


x = np.linspace(-2*np.pi, 2*np.pi, 100)

y = []
for i in x:
	y.append(np.sin(i) + i)


forecast_model = SARIMAX(y)
model_fit = forecast_model.fit(method='bfgs', disp=False)

yp = model_fit.get_forecast(100)

plt.plot(x, y)

plt.plot(x, yp.mean)

plt.show()