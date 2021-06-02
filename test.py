import os, sys
import requests
import pandas as pd
import numpy as np
import optuna
from collections import deque

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import CryptoTradingEnv
from lib import get_data, load_params


def test_model(model, env, render):
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if render:
            env.render(mode='console')
        print(info[0]['current_step'], '/', info[0]['max_steps'], end="\r", flush=True)

    return info[0]['profit']


def run_n_test(model, env, n, render=False):
    profit = []
    for i in range(n):
        profit.append(test_model(model, env, render))
        print(f'{i+1}/{n}', flush=True)
    print('Profit:', np.mean(profit), ' +/-', np.std(profit))


def load_model(policy, env, model_params):
    del model_params['policy']
    model = PPO2(policy, env, nminibatches=1, **model_params)

    if os.path.isfile(fname + '.zip'):
        model.load(fname)
    
    return model


start_time = '2021-05-20T00:00'
end_time = '2021-05-28T00:00'
max_initial_balance = 10000
episodes = 3
render = True


if __name__ == '__main__':
    fname = sys.argv[1].split('.')[0]
    policy = fname.split('-')[1]
    reward_func = fname.split('-')[2]
    granularity = int(fname.split('-')[3].strip('g'))
    coins = fname.split('-')[-1].split(',')

    data = get_data(start_time, end_time, coins, granularity)
    env_params, model_params = load_params()

    env = CryptoTradingEnv(data, coins, max_initial_balance, **env_params)
    env = make_vec_env(lambda: env, n_envs=1, vec_env_cls=DummyVecEnv)

    
    model = load_model(fname, env)

    print('Trained:')
    run_n_test(model, env, episodes, render)