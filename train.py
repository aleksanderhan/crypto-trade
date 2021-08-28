import os.path
import gym
import pandas as pd
import random
import warnings
import argparse
from time import perf_counter

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from env import CryptoTradingEnv
from test import run_n_test
from lib import get_data, load_params

warnings.filterwarnings("ignore")

device = 'cpu'
model_dir = 'model/'
algo = 'PPO'
coins = list(sorted(['algo', 'btc', 'eth', 'link']))
wiki_articles = list(sorted(['Binance', 'Bitcoin', 'Blockchain', 'Coinbase', 'Cryptocurrency', 'Ethereum']))
trend_keywords = list(sorted(['binance', 'bitcoin', 'coinbase', 'ethereum']))
start_time = '2020-01-01T00:00'
end_time = '2021-05-01T00:00'
policy = 'MlpPolicy'
granularity = 15 # Minutes
lookback_len = int(10080/granularity) # 7 days
training_iterations = 1000
epochs = 5
training_split = 0.9
n_envs = 16
env_version = CryptoTradingEnv.version


def create_env(df):
    initial_balance = random.randint(1000, 50000)
    env = CryptoTradingEnv(df, coins, initial_balance, lookback_len)
    check_env(env, warn=True)
    return env

def load_env(df, vec_norm_file, n_envs, vec_env_cls, norm_obs, norm_reward, training):
    env = make_vec_env(
        lambda: create_env(df), 
        n_envs=1,
        vec_env_cls=vec_env_cls
    )
    if os.path.isfile(vec_norm_file):
        env = VecNormalize.load(vec_norm_file, env)
        env.norm_obs = True
        env.norm_reward = False
        env.training = False
        return env
    else:
        return VecNormalize(env, norm_obs=True, norm_reward=False, training=False)

def load_model(env, model_params, fname):
    model = PPO(policy, 
                env,
                n_epochs=10,
                verbose=1,
                device=device,
                tensorboard_log='./tensorboard/',
                **model_params)

    if os.path.isfile(fname + '.zip'):
        model.load(fname)

    return model


def main():
    coins_str = ','.join(coins)
    wiki_articles_str = ','.join(wiki_articles)
    trend_keywords_str = ','.join(trend_keywords)
    experiment_name = f'PPO_env-{env_version}_p-{policy}_ll-{lookback_len}_gr-{granularity}_wpv-{wiki_articles_str}_gt-{trend_keywords_str}_c-{coins_str}'

    model_params = load_params(experiment_name)
    df = get_data(start_time, end_time, coins, wiki_articles, trend_keywords, granularity)

    slice_point = int(len(df.index) * training_split)
    train_df = df[:slice_point]
    test_df = df[slice_point:]
    test_df.reset_index(drop=True, inplace=True)

    vec_norm_file = experiment_name + '_vec_normalize.pkl'

    for i in range(training_iterations):
        start_frame = random.randint(0,  len(train_df.index) - lookback_len)
        end_frame = random.randint(start_frame+lookback_len+1, len(train_df.index))
        epoch_df = train_df[start_frame:end_frame]
        epoch_df.reset_index(drop=True, inplace=True)
        assert len(epoch_df.index) > lookback_len

        # Train model
        train_env = load_env(epoch_df, vec_norm_file, n_envs=n_envs, vec_env_cls=SubprocVecEnv, norm_obs=True, norm_reward=True, training=True)
        model = load_model(train_env, model_params, experiment_name)
        
        for e in range(epochs):
            t0 = perf_counter()
            model.learn(total_timesteps=len(epoch_df.index) - 1)
            t1 = perf_counter()
            print('iteration:', i, 'epoch:', e, 'training time:', t1 - t0)

            model.save(experiment_name)
            train_env.save(vec_norm_file)

        # Evaluate model
        validation_env = load_env(test_df, vec_norm_file, n_envs=1, vec_env_cls=DummyVecEnv, norm_obs=True, norm_reward=False, training=False)
        model = load_model(validation_env, model_params, experiment_name)

        run_n_test(model, validation_env, 5, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    if args.cuda:
        device = 'cuda'

    main()