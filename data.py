import numpy as np
import pandas as pd


coins = ['btc', 'eth']
features = ['open', 'high', 'low', 'close', 'volume', 'market_cap', 'circulating_supply' 'holding', 'keyword_freq', 'biz_sia', 'transactions']
other_features = ['account_balance']
timestamps = [1, 2]




header = []
for c in coins:
	header += [c + '_' + f for f in features]
header += other_features

print(header)


data_points = (len(coins)*len(features)) + len(other_features)
t1 = np.linspace(0, 90, data_points)
t2 = np.linspace(0, 90, data_points)

df = pd.DataFrame([t1, t2], index=timestamps, columns=header)