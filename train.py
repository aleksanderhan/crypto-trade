import os.path
import gym
import pandas as pd
import random
from time import perf_counter
import warnings

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import CryptoTradingEnv
from test import run_n_test
from lib import get_data, load_params

#warnings.filterwarnings("ignore")


coins = ['btc', 'eth'] #'link', 'ada', 'algo', 'nmr', 'xlm'] # 'FIL', 'STORJ', 'AAVE', 'COMP', 'LTC', 
coins_str = ','.join(coins)
start_time = '2021-01-01T00:00'
end_time = '2021-02-01T00:00'
epochs = 30
episodes = 1000
max_initial_balance = 50000
training_split = 0.9
n_envs=8


if __name__ == '__main__':
    env_params, model_params = load_params()
    policy = model_params['policy']
    del model_params['policy']
    
    df = get_data(start_time, end_time, coins)

    slice_point = int(len(df.index) * training_split)
    train_df = df[:slice_point]
    test_df = df[slice_point:]
    test_df.reset_index(drop=True, inplace=True)

    validation_env = make_vec_env(
        lambda: CryptoTradingEnv(test_df, coins, max_initial_balance, **env_params), 
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )

    for e in range(episodes):
        start_frame = random.randint(0, int(len(train_df.index) * training_split))
        end_frame = start_frame + int(len(train_df.index) * (1 - training_split))
        episode_df = train_df[start_frame:end_frame]
        episode_df.reset_index(drop=True, inplace=True)

        train_env = make_vec_env(
            lambda: CryptoTradingEnv(episode_df, coins, max_initial_balance, **env_params), 
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv
        )

        
        model = PPO2(policy,
                    train_env, 
                    verbose=1, 
                    noptepochs=epochs,
                    nminibatches=n_envs,
                    tensorboard_log='./tensorboard/',
                    **model_params)

        model_name = model.__class__.__name__
        reward_func = env_params['reward_func']
        fname = f'{model_name}-{policy}-{reward_func}-{coins_str}'
        
        if os.path.isfile(fname + '.zip'):
            model.load(fname)  


        t0 = perf_counter()
        for _ in range(3):
            model.learn(total_timesteps=len(episode_df.index) - 1)
        t1 = perf_counter()
        
        model.save(fname)

        print(e, 'training time:', t1 - t0)


    run_n_test(model, validation_env, 5, False)
