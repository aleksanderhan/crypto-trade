import os.path
import gym
import pandas as pd
import random
from time import perf_counter
import warnings

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.env_checker import check_env


from env import CryptoTradingEnv
from test import run_n_test
from lib import get_data, load_params


coins = ['eth']

df = get_data('2021-01-01', '2021-01-02', coins)

env_params, model_params = load_params()



train_env = CryptoTradingEnv(df, coins, 10000, **env_params)
check_env(train_env, warn=True)