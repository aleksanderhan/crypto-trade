import os.path
import gym
import requests
import pandas as pd
import random
from time import sleep, perf_counter

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from env import CryptoTradingEnv


def get_data(start_time, end_time):
    r = requests.get(f'http://127.0.0.1:5000/data?start_time={start_time}&end_time={end_time}')

    df = pd.DataFrame.from_dict(r.json())
    print(df)
    df.index = df.index.astype(int)
    return df

def get_coins():
    r = requests.get('http://127.0.0.1:5000/coins')
    return r.json()


start_time = '2020-10-01T00:00'
end_time = '2021-05-01T00:00'
frame_size = 50
epochs = 100
fname = f'model1-fs{frame_size}'
episodes = 100


if __name__ == '__main__':
    coins = get_coins()
    data = get_data(start_time, end_time)
    max_steps = len(data.index) - frame_size


    for e in range(episodes):  
        initial_balance = random.randint(1000, 20000)
        env = CryptoTradingEnv(frame_size, initial_balance, data, coins, max_steps)
        env = make_vec_env(lambda: env, n_envs=1, vec_env_cls=DummyVecEnv)


        model = PPO('MlpPolicy', env, verbose=1, n_epochs=epochs)
        if os.path.isfile(fname + '.zip'):
            print("load")
            model.load(fname)    


        t0 = perf_counter()
        model.learn(total_timesteps=max_steps)
        t1 = perf_counter()
        
        model.save(fname)

        print(e, 'training time', t1 - t0)
        