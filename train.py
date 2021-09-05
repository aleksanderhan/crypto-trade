import os.path
import gym
import pandas as pd
import random
import warnings
import argparse
import configparser
from time import perf_counter

from stable_baselines3 import PPO #, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from env import CryptoTradingEnv
from test import run_n_test
from lib import get_data, load_params, uniquefolder

warnings.filterwarnings("ignore")

device = 'cpu'
save_folder, experiment_number = next(uniquefolder('models/experiment'))

start_time = '2020-01-01T00:00' # Min: '2020-01-01T00:00'
end_time = '2020-05-01T00:00' # Max: '2021-05-01T00:00'
coins = list(sorted(['algo', 'btc', 'eth', 'link']))
wiki_articles = list(sorted(['Binance', 'Bitcoin', 'Blockchain', 'Coinbase', 'Cryptocurrency', 'Ethereum']))
trend_keywords = list(sorted(['binance', 'bitcoin', 'coinbase', 'ethereum']))

training_iterations = 1000
epochs = 5
training_split = 0.9
n_envs = 16

algo = 'PPO'
env_version = CryptoTradingEnv.version
policy = 'MlpPolicy'
granularity = 15 # Minutes
lookback_len = int(10080/granularity) # 7 days
wiki_articles_str = ','.join(wiki_articles)
trend_keywords_str = ','.join(trend_keywords)
coins_str = ','.join(coins)


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

def load_model(env, model_params, policy):
    print(model_params)
    model = PPO(policy, 
                env,
                n_epochs=10,
                verbose=1,
                device=device,
                tensorboard_log=f'./tensorboard/experiment{experiment_number}',
                **model_params)

    fname = save_folder + 'model.zip'
    if os.path.isfile(fname):
        model.load(fname)

    return model


def main(model_params):
    vec_norm_file = save_folder + 'vec_normalize.pkl'

    # Fetch data
    df = get_data(start_time, end_time, coins_str, wiki_articles_str, trend_keywords_str, granularity)

    # Split dataframe into train and test set
    slice_point = int(len(df.index) * training_split)
    train_df = df[:slice_point]
    test_df = df[slice_point:]
    test_df.reset_index(drop=True, inplace=True)

    for i in range(training_iterations):
        start_frame = random.randint(0,  len(train_df.index) - lookback_len)
        end_frame = random.randint(start_frame + lookback_len + 1, len(train_df.index))
        epoch_df = train_df[start_frame:end_frame]
        epoch_df.reset_index(drop=True, inplace=True)
        assert len(epoch_df.index) > lookback_len

        # Train model
        train_env = load_env(epoch_df, vec_norm_file, n_envs=n_envs, vec_env_cls=SubprocVecEnv, norm_obs=True, norm_reward=True, training=True)
        model = load_model(train_env, model_params, policy)
        
        for e in range(epochs):
            t0 = perf_counter()
            model.learn(total_timesteps=len(epoch_df.index) - 1)
            t1 = perf_counter()
            print('iteration:', i, 'epoch:', e, 'training time:', t1 - t0)

            model.save(save_folder + 'model.zip')
            train_env.save(vec_norm_file)

        # Evaluate model
        validation_env = load_env(test_df, vec_norm_file, n_envs=1, vec_env_cls=DummyVecEnv, norm_obs=True, norm_reward=False, training=False)
        model = load_model(validation_env, model_params, policy)

        run_n_test(model, validation_env, 5, False)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--load')
    args = parser.parse_args()

    study_name = f'experiment{experiment_number}' # TODO: hash of config file
    model_params = load_params(study_name)
    config['model_params'] = model_params
    if args.cuda:
        device = 'cuda'

    if args.load: # Read or Write configuration
        save_folder = f'models/{args.load}/'
        assert os.path.isdir(save_folder)
        config.read(save_folder + 'config.ini')

        experiment_params = config['experiment_params']
        assert experiment_params['env_version'] == env_version

        lookback_len = int(experiment_params['lookback_len'])
        policy = experiment_params['policy']
        granularity = int(experiment_params['granularity'])
        wiki_articles_str = experiment_params['wiki_articles_str']
        trend_keywords_str = experiment_params['trend_keywords_str']
        coins_str = experiment_params['coins_str']
        coins = coins_str.split(',')

        print('Continuing training', args.load)
    else:
        # Save configuration
        config['experiment_params'] = {
            'algo': algo,
            'env_version': env_version,
            'policy': policy,
            'lookback_len': lookback_len,
            'granularity': granularity,
            'wiki_articles_str': wiki_articles_str,
            'trend_keywords_str': trend_keywords_str,
            'coins_str': coins_str

        }
        os.makedirs(save_folder, exist_ok=True)
        with open(save_folder + 'config.ini', 'w') as configfile:
            config.write(configfile)

    main(model_params)
