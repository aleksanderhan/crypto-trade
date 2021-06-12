import requests
import optuna
import pandas as pd


def get_data(start_time, end_time, coins):
    coinsStr = ','.join(coins)
    r = requests.get(f'http://127.0.0.1:5000/data?start_time={start_time}&end_time={end_time}&coins={coinsStr}')

    df = pd.DataFrame.from_dict(r.json())
    print(df)
    df.index = df.index.astype(int)
    return df


def load_params(study_name):
    try:
        study = optuna.load_study(study_name=study_name, storage='sqlite:///params.db')

        params = study.best_trial.params
        print(params)

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
            'policy_kwargs': params['policy_kwargs']

        }
    except Exception as error:
        print(error)
        model_params = {
            'batch_size': 512,
            'n_steps': 512,
            'gamma': 0.90606675817696,
            'learning_rate': 0.025412921449645007,
            'gae_lambda': 0.9,
            'max_grad_norm': 0.5,
            'ent_coef': 0.0008681344845273617,
            'clip_range': 0.2025550612826041,
            'clip_range_vf': 0.28148595837701307,
            'vf_coef': 0.5,
            'policy_kwargs': None
        }

    return model_params