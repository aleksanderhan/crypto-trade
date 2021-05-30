import os
import requests
import pandas as pd
import numpy as np
from collections import deque


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from env import CryptoTradingEnv


def get_data(start_time, end_time, coins, granularity):
    coinsStr = ','.join(coins)
    r = requests.get(f'http://127.0.0.1:5000/data?start_time={start_time}&end_time={end_time}&coins={coinsStr}&granularity={granularity}')

    df = pd.DataFrame.from_dict(r.json())
    print(df)
    df.index = df.index.astype(int)
    return df
import sys

def test_model(model, env, render):
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if render:
            env.render(mode='human')
        print(info[0]['current_step'], '/', info[0]['max_steps'], end="\r", flush=True)

    return info[0]['profit']


def run_n_test(model, env, n, render=False):
    profit = []
    for i in range(n):
        profit.append(test_model(model, env, render))
        print(f'{i+1}/{n}', flush=True)
    print('Profit:', np.mean(profit), ' +/-', np.std(profit))



coins = ['btc', 'eth', 'ada', 'link', 'algo', 'nmr', 'xlm']
granularity=60
start_time = '2021-05-25T00:00'
end_time = '2021-05-28T00:00'
frame_size = 50
initial_balance = 10000
fname = 'PPO-MlpPolicy-fs50-g60-btc,eth,ada,link,algo,nmr,xlm'
policy = fname.split('-')[1]
episodes = 3
render = True


if __name__ == '__main__':
    data = get_data(start_time, end_time, coins, granularity)
    max_steps = len(data.index) - frame_size


    env = CryptoTradingEnv(frame_size, initial_balance, data, coins, fee=0, debug=False)
    #check_env(env)
    env = make_vec_env(lambda: env, n_envs=1, vec_env_cls=DummyVecEnv)


    model = PPO(policy, env, verbose=1)
    

    print('Untrained:')
    run_n_test(model, env, episodes, render)

    if os.path.isfile(fname + '.zip'):
        model.load(fname)

    print()
    print('Trained:')
    run_n_test(model, env, episodes, render)
    