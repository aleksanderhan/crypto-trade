import requests
import pandas as pd
import numpy as np
import optuna

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


from env import CryptoTradingEnv


def get_data(start_time, end_time, coins, granularity):
    coinsStr = ','.join(coins)
    r = requests.get(f'http://127.0.0.1:5000/data?start_time={start_time}&end_time={end_time}&coins={coinsStr}&granularity={granularity}')

    df = pd.DataFrame.from_dict(r.json())
    print(df)
    df.index = df.index.astype(int)
    return df


coins = ['btc']
start_time = '2021-01-01T00:00'
end_time = '2021-05-20T00:00'
training_split = 0.8
max_initial_balance = 50000
reward_func = 'sortino'

df = get_data(start_time, end_time, coins, 60)


def optimize(n_trials=5000):
    study = optuna.create_study(study_name='optimize_profit', storage='sqlite:///params.db', load_if_exists=True)
    study.optimize(objective_fn, n_trials=n_trials)


def objective_fn(trial):
    env_params = optimize_envs(trial)
    agent_params = optimize_ppo(trial)

    train_env, validation_env = initialize_envs(**env_params)
    model = PPO('MlpPolicy', train_env, device='cpu', **agent_params)
    
    model.learn(len(train_env.df))
    
    rewards, done = [], False

    obs = validation_env.reset()
    for i in range(len(validation_env.df)):
        action, _ = model.predict(obs)
        obs, reward, done, _ = validation_env.step(action)
        rewards.append(reward)
    
    return -np.mean(rewards)


def optimize_ppo(trial):
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'clip_range_vf': trial.suggest_uniform('clip_range_vf', 0., 1.)
    }


def optimize_envs(trial):
    return {
        'frame_size': int(trial.suggest_loguniform('frame_size', 10, 2000)),
    }


def initialize_envs(frame_size):
    slice_point = int(len(df.index) * training_split)
    train_df = df[:slice_point]
    test_df = df[slice_point:]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_env = CryptoTradingEnv(max_initial_balance, train_df, coins, reward_func, frame_size)
    validation_env = CryptoTradingEnv(max_initial_balance, test_df, coins, reward_func, frame_size)

    return train_env, validation_env


if __name__ == '__main__':
    optimize()