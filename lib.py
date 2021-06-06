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

        env_params = {
            'reward_len': params['reward_len'],
            'reward_func': params['reward_func']
        }
        model_params = {
            'policy': params['policy'],
            'n_steps': params['n_steps'],
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'ent_coef': params['ent_coef'],
            'cliprange': params['cliprange'],
            'lam': params['lam']
        }
    except:
        env_params = {
            'reward_len': 30,
            'reward_func': 'omega'
        }
        model_params = {
            'n_steps': 1480,
            'gamma': 0.981031411815395,
            'learning_rate': 0.077471274926592394,
            'ent_coef': 1.786525334251195e-08,
            'cliprange': 0.30701255220867096,
            'cliprange_vf': 0.30701255220867096,
            'lam': 0.9888716588160799
        }

    return env_params, model_params