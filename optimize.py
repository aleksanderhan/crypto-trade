import requests
import pandas as pd
import numpy as np
import optuna
import warnings

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv

from env import CryptoTradingEnv
from lib import get_data

warnings.filterwarnings("ignore")


coins = ['btc', 'eth']
start_time = '2021-01-01T00:00'
end_time = '2021-02-01T00:00'
policy = 'MlpLstmPolicy'
reward_func = 'sortino'
training_split = 0.8
max_initial_balance = 50000


df = get_data(start_time, end_time, coins)


def optimize(n_trials=5000):
    study = optuna.create_study(study_name=f'{policy}_{reward_func}', storage='sqlite:///params.db', load_if_exists=True)
    study.optimize(objective_fn, n_trials=n_trials)


def objective_fn(trial):
    env_params = optimize_env(trial)
    model_params = optimize_ppo2(trial)

    train_env, validation_env = initialize_envs(env_params)
    
    model = PPO2(policy, train_env, nminibatches=1, **model_params)

    train_maxlen = len(train_env.get_attr('df')[0].index) - 1
    model.learn(train_maxlen)

    rewards, done = [], False

    trades = train_env.get_attr('trades')[0]
    if len(trades) < 1:
        raise optuna.structs.TrialPruned()

    validation_maxlen = len(validation_env.get_attr('df')[0].index) - 1
    obs = validation_env.reset()
    for i in range(validation_maxlen):
        action, _states = model.predict(obs)
        obs, reward, done, _ = validation_env.step(action)
        rewards.append(reward)
        if done:
            break

    return -np.mean(rewards)


def optimize_env(trial):
    return {
        'reward_len': trial.suggest_int('reward_len', 2, 200),
        'forecast_len': trial.suggest_int('forecast_len', 2, 10),
        'confidence_interval': trial.suggest_uniform('confidence_interval', 0.7, 0.99),
        'arima_p': trial.suggest_int('arima_p', 0, 1),
        'arima_d': trial.suggest_int('arima_d', 0, 1),
        'arima_q': trial.suggest_int('arima_q', 0, 1),
        'use_forecast': trial.suggest_categorical('use_forecast', [True, False])
    }


def optimize_ppo2(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 16, 2048),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'cliprange_vf': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }


def initialize_envs(env_params):
    slice_point = int(len(df.index) * training_split)
    train_df = df[:slice_point]
    test_df = df[slice_point:]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    env_params['arima_order'] = (env_params['arima_p'], env_params['arima_d'], env_params['arima_q'])
    del env_params['arima_p']
    del env_params['arima_d']
    del env_params['arima_q']

    train_env = DummyVecEnv([lambda: CryptoTradingEnv(train_df, coins, max_initial_balance, reward_func, **env_params)])
    validation_env = DummyVecEnv([lambda: CryptoTradingEnv(test_df, coins, max_initial_balance, reward_func, **env_params)])

    return train_env, validation_env


if __name__ == '__main__':
    optimize()