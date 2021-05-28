import os.path
import gym
import requests
import pandas as pd
import random
from time import sleep, perf_counter

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from env import CryptoTradingEnv


def get_data(start_time, end_time):
    r = requests.get(f'http://127.0.0.1:5000/data?start_time={start_time}&end_time={end_time}')

    df = pd.DataFrame.from_dict(r.json())
    print(df)
    df.index = df.index.astype(int)
    return df

def get_coins():
    r = requests.get('http://127.0.0.1:5000/coins')
    return r.json()



start_time = '2021-05-18T00:00'
end_time = '2021-05-20T00:00'
frame_size = 50
epochs = 20
fname = f'model1-fs{frame_size}'
episodes = 100
max_initial_balance = 20000
training_split = 0.8



if __name__ == '__main__':
    coins = get_coins()
    df = get_data(start_time, end_time)

    slice_point = int(len(df.index) * training_split)
    train_df = df[:slice_point]
    test_df = df[slice_point:]
    test_df.reset_index(drop=True, inplace=True)

    validation_env = make_vec_env(
        lambda: CryptoTradingEnv(frame_size, max_initial_balance, test_df, coins), 
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

        model = PPO('MlpPolicy', train_env, verbose=0, n_epochs=epochs)
        if os.path.isfile(fname + '.zip'):
            model.load(fname)  


        t0 = perf_counter()
        model.learn(total_timesteps=len(episode_df.index))
        t1 = perf_counter()
        
        model.save(fname)

        mean_reward, std_reward = evaluate_policy(model, validation_env, n_eval_episodes=2, deterministic=True)
        print(e, 'training time:', t1 - t0, 'mean_reward:', mean_reward)
