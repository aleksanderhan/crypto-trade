import numpy as np
import pandas as pd




coins = ['btc', 'eth', 'link']
features = ["open", "high", "low", "close", "volume"]
timestamps = [1, 2]

num_coins = len(coins)
num_timesteps = len(timestamps)
num_features = len(features)


array = [
	[
		[1, 2, 3],
		[4, 5, 6],
	],
	[
		[7, 8, 9],
		[10, 11, 12],
	]
]
data = np.array(array)


#print(data)

names=['time', 'coin', 'features']
index = pd.MultiIndex.from_product([range(s)for s in data.shape], names=names)
df = pd.DataFrame({'data': data.flatten()}, index=index)


#print(df)
'''
                    data
time coin features      
0    0    0            1
          1            2
          2            3
     1    0            4
          1            5
          2            6
1    0    0            7
          1            8
          2            9
     1    0           10
          1           11
          2           12
'''

#print(df.unstack(level='coin'))
#print(df.unstack(level='coin').shape[1])




'''
time_vals = np.linspace(1, 50, 50)
x_vals = np.linspace(-5, 6, 12)
y_vals = np.linspace(-4, 5, 10)

measurements = np.random.rand(50,12,10)

#setup multiindex
mi = pd.MultiIndex.from_product([np.array(timestamps), np.array(coins), np.array(features)], names=['time', 'coin', 'features'])

#connect multiindex to data and save as multiindexed Series
sr_multi = pd.Series(index=mi, data=measurements.flatten())

#pull out a dataframe of x, y at time=22
#sr_multi.xs(22, level='time').unstack(level=0)

#pull out a dataframe of y, time at x=3
#sr_multi.xs(3, level='coin').unstack(level=1)

print(sr_multi)
'''