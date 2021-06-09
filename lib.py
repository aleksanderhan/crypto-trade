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
            'policy': params['policy'],
            'n_steps': params['n_steps'],
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'ent_coef': params['ent_coef'],
            'cliprange': params['cliprange'],
            'lam': params['lam']
        }
    except:
        model_params = {
            'n_steps': 2045,
            'gamma': 0.9203897188357445,
            'learning_rate': 0.08992772391199896,
            'ent_coef': 0.010952294186439615,
            'cliprange': 0.26659350900421724,
            'cliprange_vf': 0.3441675230088346,
            'lam': 0.8262543245228096
        }

    return model_params