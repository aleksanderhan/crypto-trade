import os.path
import gym
import pandas as pd
import random
from time import perf_counter
import warnings
import argparse

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from env import CryptoTradingEnv
from test import run_n_test
from lib import get_data, load_params

#warnings.filterwarnings("ignore")


device = 'cpu'
coins = list(sorted(['btc', 'eth'])) #list(sorted(['aave', 'algo', 'btc', 'comp', 'eth', 'fil', 'link', 'ltc', 'nmr', 'snx', 'uni', 'xlm', 'xtz', 'yfi']))
coins_str = ','.join(coins)
wiki_articles = list(sorted(['Bitcoin', 'Cryptocurrency', 'Ethereum']))
wiki_articles_str = ','.join(wiki_articles)
start_time = '2021-01-01T00:00'
end_time = '2021-05-31T00:00'
policy = 'MlpPolicy'
lookback_len = 4320
training_iterations = 100
epochs = 10
max_initial_balance = 50000
training_split = 0.9
n_envs = 8


def main():
    study = f'PPO_{policy}_ll{lookback_len}_{wiki_articles_str}_{coins_str}'
    model_params = load_params(study)
    
    df = get_data(start_time, end_time, coins, wiki_articles)

    slice_point = int(len(df.index) * training_split)
    train_df = df[:slice_point]
    test_df = df[slice_point:]
    test_df.reset_index(drop=True, inplace=True)

    validation_env = CryptoTradingEnv(test_df, coins, wiki_articles, max_initial_balance, lookback_len)
    #check_env(validation_env, warn=True)
    validation_env = make_vec_env(
        lambda: validation_env, 
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )

    for i in range(training_iterations):
        start_frame = random.randint(0,  int(len(train_df.index*0.9)))
        end_frame = start_frame + int(len(train_df.index*0.1))

        epochs_df = train_df[start_frame:end_frame]
        epochs_df.reset_index(drop=True, inplace=True)

        train_env = CryptoTradingEnv(epochs_df, coins, wiki_articles, max_initial_balance, lookback_len)
        #check_env(train_env, warn=True)
        train_env = make_vec_env(
            lambda: train_env, 
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv
        )
            
        model = PPO(policy, 
                    train_env,
                    n_epochs=10,
                    verbose=1,
                    device='cpu',
                    tensorboard_log='./tensorboard/',
                    **model_params)

        model_name = model.__class__.__name__
        fname = f'{model_name}-{policy}-ll{lookback_len}-{wiki_articles_str}-{coins_str}'
        if os.path.isfile(fname + '.zip'):
            model.load(fname)  

        for e in range(epochs):
            t0 = perf_counter()
            model.learn(total_timesteps=len(epochs_df.index) - 1)
            model.save(fname)
            t1 = perf_counter()

            print('iteration:', i, 'epoch:', e, 'training time:', t1 - t0)

        #train_env.close()
        model = PPO(policy,
                validation_env, 
                verbose=1,
                tensorboard_log='./tensorboard/',
                **model_params)

        if os.path.isfile(fname + '.zip'):
            model.load(fname)
        run_n_test(model, validation_env, 5, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    if args.cuda:
        device = 'cuda'

    main()