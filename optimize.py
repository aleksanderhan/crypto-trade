import requests
import pandas as pd
import numpy as np
import optuna
import warnings
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from itertools import chain, product

from env import CryptoTradingEnv
from lib import get_data, create_layers, activation, get_optuna_storage

warnings.filterwarnings("ignore")

env_version = CryptoTradingEnv.version
device = 'cpu'
coins = list(sorted(['algo', 'btc', 'eth', 'link']))
coins_str = ','.join(coins)
wiki_articles = list(sorted(['Binance', 'Bitcoin', 'Blockchain', 'Coinbase', 'Cryptocurrency', 'Ethereum']))
wiki_articles_str = ','.join(wiki_articles)
trend_keywords = list(sorted(['binance', 'bitcoin', 'coinbase', 'ethereum']))
trend_keywords_str = ','.join(trend_keywords)
start_time = '2021-01-01T00:00'
end_time = '2021-02-01T00:00'
policy = 'MlpPolicy'
training_split = 0.8
initial_balance = 5000
granularity = 15
lookback_len = int(10080/granularity)


permutations = [''.join(p) for p in chain.from_iterable(product('abc', repeat=i) for i in range(2, 4))]
permutations = list(filter(lambda nn_arch: nn_arch == ''.join(reversed(sorted(nn_arch))), permutations))


df = get_data(start_time, end_time, coins, wiki_articles, trend_keywords, granularity)


def optimize(n_trials=5000):
    study = optuna.create_study(
        study_name=f'PPO_env-{env_version}_p-{policy}_ll-{lookback_len}_gr-{granularity}_wpv-{wiki_articles_str}_gt-{trend_keywords_str}_c-{coins_str}', 
        storage=get_optuna_storage(), 
        load_if_exists=True
    )
    
    study.optimize(objective_fn, n_trials=n_trials)


def objective_fn(trial):
    model_params = sample_hyperparameters(trial)

    train_env, validation_env = initialize_envs()

    model = PPO(policy, 
                train_env,
                device=device,
                **model_params)

    train_maxlen = len(train_env.get_attr('df')[0].index) - 1
    try:
        model.learn(train_maxlen)
    except Exception as error:
        print(error)
        raise optuna.structs.TrialPruned()

    sync_envs_normalization(train_env, validation_env)
    mean_reward, _ = evaluate_policy(model, validation_env, n_eval_episodes=5)

    if mean_reward == 0:
        raise optuna.structs.TrialPruned()

    return -mean_reward


def sample_hyperparameters(trial):

    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical('n_steps', [128, 256, 512, 1024, 2048])
    gamma = trial.suggest_loguniform('gamma', 0.9, 0.9999)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1.)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.8, 1.0)
    max_grad_norm = trial.suggest_loguniform('max_grad_norm', 0.3, 5)
    clip_range = trial.suggest_uniform("clip_range", 0.1, 0.4)
    clip_range_vf = trial.suggest_uniform('clip_range_vf', 0.1, 0.4)
    vf_coef = trial.suggest_uniform('vf_coef', 0, 1)

    value_net = trial.suggest_categorical('value_net', permutations)
    policy_net = trial.suggest_categorical('policy_net', permutations)
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])

    return {
        'batch_size': batch_size,
        'n_steps': n_steps,
        'gamma': gamma,
        'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'gae_lambda': gae_lambda,
        'max_grad_norm': max_grad_norm,
        'clip_range': clip_range,
        'clip_range_vf': clip_range_vf,
        'vf_coef': vf_coef,
        'policy_kwargs': dict(
            net_arch=[dict(
                pi=create_layers(policy_net), 
                vf=create_layers(value_net)
            )],
            activation_fn=activation[activation_fn]
        )
    }


def initialize_envs():
    slice_point = int(len(df.index) * training_split)
    train_df = df[:slice_point]
    test_df = df[slice_point:]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_env = DummyVecEnv([lambda: CryptoTradingEnv(train_df, coins, initial_balance, lookback_len)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, training=True)
    validation_env = DummyVecEnv([lambda: CryptoTradingEnv(test_df, coins, initial_balance, lookback_len)])
    validation_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, training=False)

    return train_env, validation_env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    if args.cuda:
        device = 'cuda'

    optimize()