import os, sys
import requests
import pandas as pd
import numpy as np
import optuna
import warnings
import argparse
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from env import CryptoTradingEnv
from lib import get_data, load_params

warnings.filterwarnings("ignore")


def test_model(model, env, render):
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if render:
            env.render(mode='console')
        else:
            print(info[0]['current_step'], '/', info[0]['max_steps'], end="\r", flush=True)

    return info[0]['profit']


def run_n_test(model, env, n, render=False):
    profit = []
    for i in range(n):
        profit.append(test_model(model, env, render))
        print(f'{i+1}/{n}', flush=True)
    print('Profit:', np.mean(profit), ' +/-', np.std(profit))


def load_model(fname, env, model_params):
    policy = model_params['policy']
    del model_params['policy']
    model = PPO(policy, env, nminibatches=1, **model_params)

    if os.path.isfile(fname + '.zip'):
        model.load(fname)
    
    return model


start_time = '2021-01-01T00:00'
end_time = '2021-02-01T00:00'
max_initial_balance = 10000
episodes = 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", action="store_true")
    parser.add_argument('fname')
    args = parser.parse_args()
    print(args)

    fname = args.fname
    policy = fname.split('-')[1]
    reward_func = fname.split('-')[2]
    coins = fname.split('-')[-1].split(',')

    data = get_data(start_time, end_time, coins)
    env_params, model_params = load_params()

    env = CryptoTradingEnv(data, coins, max_initial_balance, **env_params)
    env = make_vec_env(lambda: env, n_envs=1, vec_env_cls=DummyVecEnv)

    
    model = load_model(fname, env, model_params)

    #mean_reward, std_reward = evaluate_policy(model, env, deterministic=False, render=args.r, n_eval_episodes=episodes)
    #print('mean_reward:', mean_reward, 'std_reward', std_reward)

    run_n_test(model, env, episodes, args.r)