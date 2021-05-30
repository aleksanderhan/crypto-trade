import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

coins = ['btc', 'eth', 'ada']


data = {
	'btc': np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
	'eth': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
	'ada': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
	}


fig = plt.figure()

num_rows = 2
num_columns = int(np.ceil((len(coins) + 1)/num_rows))

print('num_rows:', num_rows, 'num_columns', num_columns)

net_worth_ax = plt.subplot2grid((num_rows, num_columns), (0, 0), rowspan=1, colspan=1)


price_axes = {}


for i, coin in enumerate(coins):
	x = (i+1)%num_rows
	y = 
	print('x,y=', x,y)
	price_ax = plt.subplot2grid((num_rows, num_columns), (x, y), rowspan=1, colspan=1)
	price_axes[coin] = price_ax





plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0.2)


plt.show(block=True)