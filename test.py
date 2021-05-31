import os, sys
import requests
import pandas as pd
import numpy as np
import optuna
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

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



start_time = '2021-05-20T00:00'
end_time = '2021-05-28T00:00'
max_initial_balance = 10000
episodes = 3
render = True


if __name__ == '__main__':
    fname = sys.argv[1].split('.')[0]
    policy = fname.split('-')[1]
    reward_func = fname.split('-')[2]
    frame_size = int(fname.split('-')[3].strip('fs'))
    granularity = int(fname.split('-')[4].strip('g'))
    coins = fname.split('-')[-1].split(',')

    data = get_data(start_time, end_time, coins, granularity)

    _, model_params = load_params()


    env = CryptoTradingEnv(data, coins, max_initial_balance, reward_func, frame_size)
    #check_env(env)
    env = make_vec_env(lambda: env, n_envs=1, vec_env_cls=DummyVecEnv)


    model = PPO(policy, env, **model_params)
    

    #print('Untrained:')
    #run_n_test(model, env, episodes, render)

    if os.path.isfile(fname + '.zip'):
        model.load(fname)

    print()
    print('Trained:')
    run_n_test(model, env, episodes, render)

    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
    #print('mean_reward:', mean_reward, 'std_reward:', std_reward)