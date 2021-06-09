import os, sys
import requests
import pandas as pd
import numpy as np
import optuna
import warnings
import argparse
from collections import deque

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.evaluation import evaluate_policy

from env import CryptoTradingEnv
from lib import get_data, load_params

warnings.filterwarnings("ignore")


def test_model(model, env, render):
    obs = env.reset()
    done = False

    total_reward = 0
    while not done:
        if render:
            env.render(mode=render)
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if not render:
            print(info[0]['current_step'], '/', info[0]['max_steps'], end="\r", flush=True)
    print()

    return info[0]['profit'], total_reward


def run_n_test(model, env, n, render=False):
    profits = []
    total_rewards = []
    for i in range(n):
        profit, total_reward = test_model(model, env, render)
        profits.append(profit)
        total_rewards.append(total_reward)
        print(f'{i+1}/{n}', flush=True)
    print('Profit:', np.mean(profits), ' +/-', np.std(profits))
    print('Total reward:', np.mean(total_rewards), '+/-', np.std(total_rewards))


def load_model(fname, policy, env, model_params):
    model = PPO2(policy, env, nminibatches=1, **model_params)

    if os.path.isfile(fname + '.zip'):
        model.load(fname)
    
    return model


start_time = '2021-06-01T00:00'
end_time = '2021-06-06T00:00'
max_initial_balance = 10000
episodes = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r")
    parser.add_argument('fname')
    args = parser.parse_args()
    print(args)

    fname = args.fname.split('.')[0]
    model_name = fname.split('-')[0]
    policy = fname.split('-')[1]
    coins = fname.split('-')[-1].split(',')
    coins_str = ','.join(sorted(coins))


    data = get_data(start_time, end_time, coins)

    study_name = f'{model_name}_{policy}_{coins_str}'
    model_params = load_params(study_name)

    env = CryptoTradingEnv(data, coins, max_initial_balance)
    env = make_vec_env(lambda: env, n_envs=1, vec_env_cls=DummyVecEnv)

    
    model = load_model(fname, policy, env, model_params)

    #mean_reward, std_reward = evaluate_policy(model, env, deterministic=False, render=bool(args.r), n_eval_episodes=episodes)
    #print('mean_reward:', mean_reward, 'std_reward', std_reward)

    run_n_test(model, env, episodes, args.r)