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


def get_data(frame_size, start_time, end_time):
    r = requests.get(f'http://127.0.0.1:5000/data?frame_size={frame_size}&start_time={start_time}&end_time={end_time}') # Use frame_size to get frame_size amount of datapoints

    df = pd.DataFrame.from_dict(r.json())
    print(df)
    df.index = df.index.astype(int)
    return df

def get_coins():
    r = requests.get('http://127.0.0.1:5000/coins')
    return r.json()


fname = 'model0'
start_time = '2021-05-16T00:00'
end_time = '2021-05-20T00:00'
frame_size = 5
epochs = 1
data = get_data(frame_size, start_time, end_time)
max_steps = len(data.index) - frame_size
initial_balance = random.randint(1000, 20000)


# multiprocess environment
env = CryptoTradingEnv(frame_size, initial_balance, data, get_coins(), max_steps)
env = make_vec_env(lambda: env, n_envs=1, vec_env_cls=DummyVecEnv)


model = PPO('MlpPolicy', env, verbose=0)
if os.path.isfile(fname + '.zip'):
    model.load(fname)    


t0 = perf_counter()
for e in range(epochs):
    model.learn(total_timesteps=max_steps)
    model.save(fname)
t1 = perf_counter()



obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render(mode='console')
    if done.all():
        print('Episode finished after {} timesteps'.format(info[0]['current_step']))
        break;
        #obs = env.reset()



mean_reward_random, std_reward_random = evaluate_policy(PPO('MlpPolicy', env, verbose=0), env, n_eval_episodes=4, deterministic=True)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=4, deterministic=True)
print('mean_reward_random', mean_reward_random)
print('std_reward_random', std_reward_random)
print('mean_reward', mean_reward)
print('std_reward', std_reward)
print('training time', t1 - t0)