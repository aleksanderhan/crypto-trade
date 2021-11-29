import os, sys
import requests
import pandas as pd
import numpy as np
import optuna
import warnings
import argparse
import configparser
from collections import deque

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from env import CryptoTradingEnv
from lib import get_data, load_params

warnings.filterwarnings("ignore")

start_time = '2020-01-01T00:00' #'2021-05-01T00:00'
end_time = '2020-02-01T00:00' #'2021-06-12T00:00'
initial_balance = 10000
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
        total_reward += reward[0]
        if not render:
            print(info[0]['current_step'], '/', env.get_attr('max_steps')[0], end="\r", flush=True)
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
    parts =  os.path.normpath(args.model_folder).split(os.path.sep)
    study_name = f'{parts[-1]}'

    config = configparser.ConfigParser()
    config.read(args.model_folder + 'config.ini')
    experiment_params = config['experiment_params']

    lookback_len = int(experiment_params['lookback_len'])
    policy = experiment_params['policy']
    granularity = int(experiment_params['granularity'])
    wiki_articles_str = experiment_params['wiki_articles_str']
    trend_keywords_str = experiment_params['trend_keywords_str']
    coins_str = experiment_params['coins_str']
    coins = coins_str.split(',')

    df = get_data(start_time, end_time, coins_str, wiki_articles_str, trend_keywords_str, granularity)

    model_params = load_params(study_name)

    env = CryptoTradingEnv(df, coins, initial_balance, lookback_len, fee=0.001)
    assert env.version == experiment_params['env_version']
    env = make_vec_env(
        lambda: env, 
        n_envs=1, 
        vec_env_cls=DummyVecEnv
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)

    vec_norm_file = args.model_folder + 'vec_normalize.pkl'
    if os.path.isfile(vec_norm_file):
        env = VecNormalize.load(vec_norm_file, env)
        env.norm_obs = True
        env.norm_reward = False
        env.training = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)

    model = PPO(policy,
                env, 
                verbose=1,
                tensorboard_log='./tensorboard/',
                **model_params)

    if os.path.isfile(args.model_folder + 'model.zip'):
        model.load(args.model_folder + 'model')

    #mean_reward, std_reward = evaluate_policy(model, env, deterministic=False, render=bool(args.r), n_eval_episodes=episodes)
    #print('mean_reward:', mean_reward, 'std_reward', std_reward)

    run_n_test(model, env, episodes, args.r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r')
    parser.add_argument('model_folder')
    args = parser.parse_args()
    print(args)

    main(args)

    