import os.path
import gym
import requests
import pandas as pd
import random
from time import sleep, perf_counter

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from env import CryptoTradingEnv


def get_data(frame_size, start_time, end_time):
    r = requests.get(f'http://127.0.0.1:5000/data?frame_size={frame_size}&start_time={start_time}&end_time={end_time}') # Use frame_size to get frame_size amount of datapoints

    df = pd.DataFrame.from_dict(r.json())
    print(df)
    df.index = df.index.astype(int)
    return df

def get_coins():
    r = requests.get('http://127.0.0.1:5000/coins')
    return r.json()




fname = "model0"
frame_size = 5
initial_balance = random.randint(1000, 20000)
start_time = '2021-05-20T00:00'
end_time = '2021-05-24T00:00'
max_steps = 100
epochs = 10

# multiprocess environment
env = CryptoTradingEnv(frame_size, initial_balance, get_data(frame_size, start_time, end_time), get_coins(), max_steps)
env = make_vec_env(lambda: env, n_envs=1)
print(env)


if os.path.isfile(fname + '.zip'):
    model = PPO.load(fname)
else:
    model = PPO('MlpPolicy', env, verbose=1)



for e in range(epochs):
    model.learn(total_timesteps=max_steps)
    model.save(fname)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=4, deterministic=True)
    env.reset()
    print('mean_reward', mean_reward)
    print('std_reward', std_reward)


obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done.all():
        print('Episode finished after {} timesteps'.format(info[0]['current_step']))
        break;
        #obs = env.reset()