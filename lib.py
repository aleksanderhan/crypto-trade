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
            'n_steps': 476,
            'gamma': 0.90606675817696,
            'learning_rate': 0.025412921449645007,
            'ent_coef': 0.0008681344845273617,
            'cliprange': 0.2025550612826041,
            'cliprange_vf': 0.28148595837701307,
            'lam': 0.904350962253066
        }

    return model_params