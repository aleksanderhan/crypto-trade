import gym
import requests
import pandas as pd
from time import sleep

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from env import CryptoTradingEnv


def get_data():
    r = requests.get('http://127.0.0.1:5000/data') # Use frame_size to get frame_size amount of datapoints
    df = pd.DataFrame.from_dict(r.json())
    df.index = df.index.astype(int)
    return df

def get_coins():
    r = requests.get('http://127.0.0.1:5000/coins')
    return r.json()




frame_size = 5
initial_balance = 5000
max_steps = 200


# multiprocess environment
env = CryptoTradingEnv(frame_size, initial_balance, get_data(), get_coins(), max_steps)
env = make_vec_env(lambda: env, n_envs=1)
print(env)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=max_steps)
model.save("model0")

#del model # remove to demonstrate saving and loading
#model = PPO2.load("model0")



obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    #env.render()
    if done.all():
        print("Episode finished after {} timesteps".format(info[0]['current_step']))
        env.render()
        obs = env.reset()
