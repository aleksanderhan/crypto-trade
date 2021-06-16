import os.path
import gym
import pandas as pd
import random
import warnings
import argparse
from time import perf_counter

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from env import CryptoTradingEnv
from test import run_n_test
from lib import get_data, load_params

warnings.filterwarnings("ignore")

device = 'cpu'
model_dir = 'model/'
algo = 'PPO'
coins = list(sorted(['btc', 'eth'])) #list(sorted(['aave', 'algo', 'btc', 'comp', 'eth', 'fil', 'link', 'ltc', 'nmr', 'snx', 'uni', 'xlm', 'xtz', 'yfi']))
wiki_articles = list(sorted(['Bitcoin', 'Cryptocurrency', 'Ethereum']))
start_time = '2021-06-01T00:00'
end_time = '2021-06-08T00:00'
policy = 'MlpPolicy'
lookback_len = 4320 # 3 days
training_iterations = 100
epochs = 10
max_initial_balance = 50000
training_split = 0.9
n_envs = 8


def create_env(df):
    env = CryptoTradingEnv(df, coins, wiki_articles, max_initial_balance, lookback_len)
    check_env(env, warn=True)
    return env

def load_env(df, vec_norm_file, n_envs, vec_env_cls, norm_obs, norm_reward, training):
    env = make_vec_env(
        lambda: create_env(df), 
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    if os.path.isfile(vec_norm_file):
        env = VecNormalize.load(vec_norm_file, env)
        env.norm_obs = True
        env.norm_reward = False
        env.training = False
        return env
    else:
        return VecNormalize(env, norm_obs=True, norm_reward=False, training=False)

def load_model(env, model_params, model_file):
    model = PPO(policy, 
                env,
                n_epochs=10,
                verbose=1,
                device=device,
                tensorboard_log='./tensorboard/',
                **model_params)

    if os.path.isfile(model_file + '.zip'):
        model.load(model_file)

    return model


def main():
    coins_str = ','.join(coins)
    wiki_articles_str = ','.join(wiki_articles)
    study = f'PPO_{policy}_ll{lookback_len}_{wiki_articles_str}_{coins_str}'

    model_params = load_params(study)
    model_params['n_steps'] = 512
    df = get_data(start_time, end_time, coins, wiki_articles)

    slice_point = int(len(df.index) * training_split)
    train_df = df[:slice_point]
    test_df = df[slice_point:]
    test_df.reset_index(drop=True, inplace=True)

    model_file = f'{algo}-{policy}-ll{lookback_len}-{wiki_articles_str}-{coins_str}'
    vec_norm_file = model_file + '_vec_normalize.pkl'

    for i in range(training_iterations):
        start_frame = random.randint(0,  int(len(train_df.index*0.9)))
        end_frame = start_frame + int(len(train_df.index*0.1))
        epoch_df = train_df[start_frame:end_frame]
        epoch_df.reset_index(drop=True, inplace=True)

        # Train model
        train_env = load_env(epoch_df, vec_norm_file, n_envs=n_envs, vec_env_cls=SubprocVecEnv, norm_obs=True, norm_reward=True, training=True)
        model = load_model(train_env, model_params, model_file)
        
        for e in range(epochs):
            t0 = perf_counter()
            model.learn(total_timesteps=len(epoch_df.index) - 1)
            t1 = perf_counter()
            print('iteration:', i, 'epoch:', e, 'training time:', t1 - t0)

            model.save(model_file)
            train_env.save(vec_norm_file)

        # Evaluate model
        validation_env = load_env(test_df, vec_norm_file, n_envs=1, vec_env_cls=DummyVecEnv, norm_obs=True, norm_reward=False, training=False)
        model = load_model(validation_env, model_params, model_file)

        run_n_test(model, validation_env, 5, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    if args.cuda:
        device = 'cuda'

    main()