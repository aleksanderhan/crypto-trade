import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

coins = ['btc', 'eth']


data = {
	'btc': np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
	'eth': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	}


fig = plt.figure()


net_worth_ax = plt.subplot2grid((10, 1), (0, 0), rowspan=2, colspan=1)


plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0)


plt.show(block=False)