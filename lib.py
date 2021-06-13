import requests
import optuna
import pandas as pd

from torch import nn as nn


hidden_units = {
    'a': 64,
    'b': 128,
    'c': 256,
    'd': 512
}

activation = {
    'tanh': nn.Tanh, 
    'relu': nn.ReLU, 
    'elu': nn.ELU, 
    'leaky_relu': nn.LeakyReLU
}


def get_data(start_time, end_time, coins, wiki_articles):
    coins_str = ','.join(coins)
    wiki_articles_str = ','.join(wiki_articles)
    r = requests.get(f'http://127.0.0.1:5000/data?start_time={start_time}&end_time={end_time}&coins={coins_str}&wiki_articles={wiki_articles_str}')

    df = pd.DataFrame.from_dict(r.json())
    print(df)
    df.index = df.index.astype(int)
    return df


def create_layers(permutation):
    return [hidden_units[i] for i in permutation]


def load_params(study_name):
    try:
        study = optuna.load_study(study_name=study_name, storage='sqlite:///params.db')
        params = study.best_trial.params

        model_params = {
            'batch_size': params['batch_size'],
            'n_steps': params['n_steps'],
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'ent_coef': params['ent_coef'],
            'gae_lambda': params['gae_lambda'],
            'max_grad_norm': params['max_grad_norm'],
            'clip_range': params['clip_range'],
            'clip_range_vf': params['clip_range_vf'],
            'vf_coef': params['vf_coef'],
            'policy_kwargs': dict(
                net_arch=[dict(
                    pi=create_layers(params['policy_net']), 
                    vf=create_layers(params['value_net'])
                )],
                activation_fn=activation[params['activation_fn']]
            )
        }
    except Exception as error:
        print(error)
        model_params = {
            'batch_size': 512,
            'n_steps': 512,
            'gamma': 0.963,
            'learning_rate': 0.0024,
            'gae_lambda': 0.831,
            'max_grad_norm': 3.71,
            'ent_coef': 1.06288e-07,
            'clip_range':  0.227,
            'clip_range_vf': 0.154,
            'vf_coef': 0.161,
            'policy_kwargs': dict(
                net_arch=[dict(
                    pi=create_layers('baa'), 
                    vf=create_layers('cac')
                )],
                activation_fn=activation['relu']
            )
        }

    print('loading params:', model_params)
    return model_params
