import requests
import pandas as pd
import numpy as np
import optuna
import warnings

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from torch import nn as nn

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
    model = PPO(policy, 
                train_env,
                device='cpu',
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

    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical('n_steps', [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_loguniform('gamma', 0.9, 0.9999)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1.)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical('max_grad_norm', [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    clip_range_vf = trial.suggest_uniform('cliprange_vf', 0.1, 0.4)
    vf_coef = trial.suggest_uniform('vf_coef', 0, 1)

    net_arch = trial.suggest_categorical('net_arch', ['small', 'medium', 'large'])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])

    net_arch = {
        'small': [dict(pi=[64, 64], vf=[64, 64])],
        'medium': [dict(pi=[256, 256], vf=[256, 256])],
        'large': [dict(pi=[512, 512], vf=[512, 512])]
    }[net_arch]

    activation_fn = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'elu': nn.ELU, 'leaky_relu': nn.LeakyReLU}[activation_fn]

    return {
        'batch_size': batch_size,
        'n_steps': n_steps,
        'gamma': gamma,
        'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'gae_lambda': gae_lambda,
        'max_grad_norm': max_grad_norm,
        'clip_range': clip_range,
        'clip_range_vf': clip_range_vf,
        'vf_coef': vf_coef,
        'policy_kwargs': dict(
            net_arch=net_arch,
            activation_fn=activation_fn
        )
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