import requests
import pandas as pd
import numpy as np
import optuna
import warnings

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from env import CryptoTradingEnv
from lib import get_data

warnings.filterwarnings("ignore")


coins = ['aave', 'algo', 'btc', 'comp', 'eth', 'fil', 'link', 'ltc', 'nmr', 'snx', 'uni', 'xlm', 'xtz', 'yfi']
coins_str = ','.join(sorted(coins))
start_time = '2021-01-01T00:00'
end_time = '2021-02-01T00:00'
policy = 'MlpPolicy'
training_split = 0.8
max_initial_balance = 50000
lookback_len = 1440


df = get_data(start_time, end_time, coins)


def optimize(n_trials=5000):
    study = optuna.create_study(study_name=f'PPO_{policy}_ll{lookback_len}_{coins_str}', storage='sqlite:///params.db', load_if_exists=True)
    study.optimize(objective_fn, n_trials=n_trials)


def objective_fn(trial):
    model_params = optimize_ppo(trial)

    train_env, validation_env = initialize_envs()  
    model = PPO(
        policy, 
        train_env, 
        batch_size=model_params['n_steps'], # https://github.com/DLR-RM/stable-baselines3/issues/440
        **model_params)

    train_maxlen = len(train_env.get_attr('df')[0].index) - 1
    model.learn(train_maxlen)

    trades = train_env.get_attr('trades')[0]
    if len(trades) < 1:
        raise optuna.structs.TrialPruned()


    mean_reward, _ = evaluate_policy(model, validation_env, n_eval_episodes=5)

    if mean_reward == 0:
        raise optuna.structs.TrialPruned()

    return -mean_reward


def optimize_ppo(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 16, 2048),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'clip_range': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'clip_range_vf': trial.suggest_uniform('cliprange_vf', 0.1, 0.4)
    }


def initialize_envs():
    slice_point = int(len(df.index) * training_split)
    train_df = df[:slice_point]
    test_df = df[slice_point:]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_env = DummyVecEnv([lambda: CryptoTradingEnv(train_df, coins, max_initial_balance)])
    validation_env = DummyVecEnv([lambda: CryptoTradingEnv(test_df, coins, max_initial_balance)])

    return train_env, validation_env


if __name__ == '__main__':
    optimize()