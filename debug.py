import os, sys
import requests
import pandas as pd
import numpy as np
import optuna
from collections import deque

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import CryptoTradingEnv
from lib import get_data, load_params


start_time = '2021-05-20T00:00'
end_time = '2021-05-28T00:00'
max_initial_balance = 10000


env = CryptoTradingEnv(data, coins, max_initial_balance, **env_params)
    env = make_vec_env(lambda: env, n_envs=1, vec_env_cls=DummyVecEnv)


model = PPO2(MlpLstmPolicy, env, nminibatches=1, **model_params)
