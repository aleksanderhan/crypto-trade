import os.path
import gym
import requests
import pandas as pd
import random
from time import perf_counter

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from env import CryptoTradingEnv
from test import run_n_test


def get_data(start_time, end_time, coins, granularity):
    coinsStr = ','.join(coins)
    r = requests.get(f'http://127.0.0.1:5000/data?start_time={start_time}&end_time={end_time}&coins={coinsStr}&granularity={granularity}')

    df = pd.DataFrame.from_dict(r.json())
    print(df)
    df.index = df.index.astype(int)
    return df


coins = ['btc', 'eth', 'ada', 'link', 'algo', 'nmr', 'xlm'] # 'FIL', 'STORJ', 'AAVE', 'COMP', 'LTC', 
coins_str = ','.join(coins)
policy='MlpPolicy'
granularity = 60
start_time = '2021-01-01T00:00'
end_time = '2021-05-20T00:00'
frame_size = 100
epochs = 20
episodes = 1000
max_initial_balance = 50000
training_split = 0.8



if __name__ == '__main__':

    df = get_data(start_time, end_time, coins, granularity)

    slice_point = int(len(df.index) * training_split)
    train_df = df[:slice_point]
    test_df = df[slice_point:]
    test_df.reset_index(drop=True, inplace=True)

    test_env = CryptoTradingEnv(frame_size, max_initial_balance, test_df, coins)
    #check_env(test_env)

    validation_env = make_vec_env(
        lambda: test_env, 
        n_envs=1, 
        vec_env_cls=DummyVecEnv
    )

    for e in range(episodes):
        start_frame = random.randint(0, int(len(train_df.index) * training_split))
        end_frame = start_frame + int(len(train_df.index) * (1 - training_split))
        episode_df = train_df[start_frame:end_frame]
        episode_df.reset_index(drop=True, inplace=True)

        train_env = make_vec_env(
            lambda: CryptoTradingEnv(frame_size, max_initial_balance, episode_df, coins), 
            n_envs=1, 
            vec_env_cls=DummyVecEnv
        )

        model = PPO(policy, 
                    train_env, 
                    verbose=1, 
                    n_epochs=epochs,
                    device='cpu',
                    tensorboard_log='./tensorboard/')

        model_name = model.__class__.__name__
        fname = f'{model_name}-{policy}-fs{frame_size}-g{granularity}-{coins_str}'
        
        if os.path.isfile(fname + '.zip'):
            model.load(fname)  


        t0 = perf_counter()
        model.learn(total_timesteps=len(episode_df.index))
        t1 = perf_counter()
        
        model.save(fname)

        print(e, 'training time:', t1 - t0)

        run_n_test(model, validation_env, 5, render)
