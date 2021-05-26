import requests
import pandas as pd
import os


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from env import CryptoTradingEnv


def get_data(frame_size, start_time, end_time):
    r = requests.get(f'http://127.0.0.1:5000/data?frame_size={frame_size}&start_time={start_time}&end_time={end_time}') # Use frame_size to get frame_size amount of datapoints
    
    #print(r.json())

    df = pd.DataFrame.from_dict(r.json())
    print(df)
    df.index = df.index.astype(int)
    return df

def get_coins():
    r = requests.get('http://127.0.0.1:5000/coins')
    return r.json()




fname = 'model1-fs50'
start_time = '2021-05-01T00:00'
end_time = '2021-05-20T00:00'
frame_size = 50
epochs = 10
initial_balance = 10000


if __name__ == '__main__':
    data = get_data(frame_size, start_time, end_time)
    max_steps = len(data.index) - frame_size


    env = CryptoTradingEnv(frame_size, initial_balance, data, get_coins(), max_steps)
    env = make_vec_env(lambda: env, n_envs=1, vec_env_cls=DummyVecEnv)


    model = PPO('MlpPolicy', env, verbose=0, n_epochs=epochs)
    if os.path.isfile(fname + '.zip'):
        model.load(fname) 


    obs = env.reset()
    print(obs)

    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render(mode='console')
        if done.all():
            #env.render(mode='console')
            print('Episode finished after {} timesteps'.format(info[0]['current_step']))
            break;


    mean_reward_random, std_reward_random = evaluate_policy(PPO('MlpPolicy', env, verbose=0), env, n_eval_episodes=10, deterministic=True)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print('              trained         random')
    print('mean_reward', mean_reward, mean_reward_random)
    print('std_reward', std_reward, std_reward_random)
    print()
    print('delta_mean_reward', mean_reward - mean_reward_random)
    print('delta_std_reward', std_reward - std_reward_random)