import requests
import pandas as pd
import os
from collections import deque


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from env import CryptoTradingEnv


def get_data(start_time, end_time, coins, granularity):
    coinsStr = ','.join(coins)
    r = requests.get(f'http://127.0.0.1:5000/data?start_time={start_time}&end_time={end_time}&coins={coinsStr}&granularity={granularity}')

    df = pd.DataFrame.from_dict(r.json())
    print(df)
    df.index = df.index.astype(int)
    return df




coins = ['btc', 'eth', 'ada', 'link', 'algo', 'nmr', 'xlm']
granularity=60
start_time = '2021-05-20T00:00'
end_time = '2021-05-28T00:00'
frame_size = 50
epochs = 10
initial_balance = 10000
fname = 'model0-fs50-g60-btc,eth,ada,link,algo,nmr,xlm'


if __name__ == '__main__':
    data = get_data(start_time, end_time, coins, granularity)
    max_steps = len(data.index) - frame_size


    env = CryptoTradingEnv(frame_size, initial_balance, data, coins, debug=False)
    check_env(env)
    env = make_vec_env(lambda: env, n_envs=1, vec_env_cls=DummyVecEnv)


    model = PPO('MlpPolicy', env, verbose=0, n_epochs=epochs)
    if os.path.isfile(fname + '.zip'):
        model.load(fname) 


    obs = env.reset()
    profit = deque(maxlen=2)
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        profit.append(env.render(mode='console'))
        if done.all():
            print(info[0]['trades'])
            print(f'Profit: {profit}')
            print('Episode finished after {} timesteps'.format(info[0]['current_step']))
            #obs = env.reset()
            break;

            
    mean_reward_random, std_reward_random = evaluate_policy(PPO('MlpPolicy', env, verbose=0), env, n_eval_episodes=10, deterministic=True)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print('              trained         random')
    print('mean_reward', mean_reward, mean_reward_random)
    print('std_reward', std_reward, std_reward_random)
    print()
    print('delta_mean_reward', mean_reward - mean_reward_random)
    print('delta_std_reward', std_reward - std_reward_random)
    