from flask import Flask
import numpy as np
import pandas as pd
import json

coins = ['btc', 'eth']
features = ['open', 'high', 'low', 'close', 'volume'] # 'market_cap', 'circulating_supply', 'keyword_freq', 'biz_sia', 'transactions'
other_features = []

def create_header():
	header = []
	for c in coins:
		header += [c + '_' + f for f in features]
	header += other_features
	return header

header = create_header()

timesteps = 10
timestamps = list(range(timesteps))
num_features = (len(coins)*len(features)) + len(other_features)

data = [np.linspace(0, 100, num_features) for _ in range(timesteps)]


df = pd.DataFrame(data, index=timestamps, columns=header)
print(df)


app = Flask(__name__)

@app.route('/data')
def get_data():
    return df.to_json()

@app.route('/coins')
def get_coins():
    return json.dumps(coins)