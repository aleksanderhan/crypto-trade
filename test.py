import os, sys
import requests
import pandas as pd
import numpy as np
import optuna
import warnings
import argparse
from collections import deque


from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from env import CryptoTradingEnv
from lib import get_data, load_params

#warnings.filterwarnings("ignore")


start_time = '2021-06-01T00:00'
end_time = '2021-06-13T00:00'
max_initial_balance = 10000
episodes = 1


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


def main(args):
    fname = args.fname.split('.')[0]
    model_name = fname.split('-')[0]
    policy = fname.split('-')[1]
    lookback_len = int(fname.split('-')[2].strip('ll'))
    wiki_articles_str = fname.split('-')[3]
    wiki_articles = wiki_articles_str.split(',')
    coins_str = fname.split('-')[-1]
    coins = coins_str.split(',')

    df = get_data(start_time, end_time, coins, wiki_articles)

    study_name = f'{model_name}_{policy}_ll{lookback_len}_{wiki_articles_str}_{coins_str}'
    model_params = load_params(study_name)

    env = make_vec_env(
        lambda: CryptoTradingEnv(df, coins, wiki_articles, max_initial_balance, lookback_len), 
        n_envs=1, 
        vec_env_cls=DummyVecEnv
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)

    model = PPO(policy,
                env, 
                verbose=1,
                tensorboard_log='./tensorboard/',
                **model_params)

    model_file = '/model/' + fname
    vec_norm_file = model_file + '_vec_normalize.pkl'
    if os.path.isfile(model_file + '.zip'):
        model.load(model_file)
    if os.path.isfile(vec_norm_file):
        train_env.load(vec_norm_file)

    #mean_reward, std_reward = evaluate_policy(model, env, deterministic=False, render=bool(args.r), n_eval_episodes=episodes)
    #print('mean_reward:', mean_reward, 'std_reward', std_reward)

    run_n_test(model, env, episodes, args.r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r")
    parser.add_argument('fname')
    args = parser.parse_args()
    print(args)

    main(args)

    