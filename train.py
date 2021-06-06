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

warnings.filterwarnings("ignore")


coins = ['aave', 'algo', 'btc', 'comp', 'eth', 'fil', 'link', 'ltc', 'nmr', 'snx', 'uni', 'xlm', 'yfi']
coins_str = ','.join(sorted(coins))
start_time = '2020-01-01T00:00'
end_time = '2021-05-31T00:00'
policy = 'MlpLstmPolicy'
training_iterations = 100
epochs = 100
max_initial_balance = 50000
training_split = 0.9
n_envs = 4


if __name__ == '__main__':
    study = f'{policy}_{coins_str}'
    env_params, model_params = load_params(study)
    print('env_params', env_params)
    print('model_params', model_params)
    
    df = get_data(start_time, end_time, coins)

    slice_point = int(len(df.index) * training_split)
    train_df = df[:slice_point]
    test_df = df[slice_point:]
    test_df.reset_index(drop=True, inplace=True)

    validation_env = CryptoTradingEnv(test_df, coins, max_initial_balance, **env_params)
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

        train_env = CryptoTradingEnv(epochs_df, coins, max_initial_balance, **env_params)
        #check_env(train_env, warn=True)
        train_env = make_vec_env(
            lambda: train_env, 
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv
        )
            
        model = PPO2(policy,
                    train_env, 
                    verbose=2, 
                    noptepochs=10,
                    nminibatches=n_envs,
                    tensorboard_log='./tensorboard/',
                    **model_params)

        model_name = model.__class__.__name__
        fname = f'{model_name}-{policy}-{coins_str}'
        if os.path.isfile(fname + '.zip'):
            model.load(fname)  

        for e in range(epochs):
            t0 = perf_counter()
            model.learn(total_timesteps=len(epochs_df.index) - 1)
            model.save(fname)
            t1 = perf_counter()

            print('iteration:', i, 'epoch:', e, 'training time:', t1 - t0)


        run_n_test(model, validation_env, 5, False)
