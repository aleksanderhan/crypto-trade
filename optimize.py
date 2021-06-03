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
training_split = 0.8
max_initial_balance = 50000


df = get_data(start_time, end_time, coins, 60)


def optimize(n_trials=5000):
    study = optuna.create_study(study_name='optimize_profit', storage='sqlite:///params.db', load_if_exists=True)
    study.optimize(objective_fn, n_trials=n_trials)


def objective_fn(trial):
    env_params = optimize_env(trial)
    model_params = optimize_ppo2(trial)

    train_env, validation_env = initialize_envs(env_params)

    policy = model_params['policy']
    del model_params['policy']
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
        'reward_func': trial.suggest_categorical('reward_func', ['sortino', 'calmar', 'omega', 'simple']),
        'reward_len': trial.suggest_int('reward_len', 2, 200),
        'forecast_len': trial.suggest_int('forecast_len', 1, 10),
        'lookback_interval': trial.suggest_int('lookback_interval', 10, 50),
        'confidence_interval': trial.suggest_uniform('confidence_interval', 0.7, 0.99)
    }


def optimize_ppo2(trial):
    return {
        'policy': trial.suggest_categorical('policy', ['MlpPolicy', 'MlpLstmPolicy', 'MlpLnLstmPolicy']),
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

    train_env = DummyVecEnv([lambda: CryptoTradingEnv(train_df, coins, max_initial_balance, **env_params)])
    validation_env = DummyVecEnv([lambda: CryptoTradingEnv(test_df, coins, max_initial_balance, **env_params)])

    return train_env, validation_env


if __name__ == '__main__':
    optimize()