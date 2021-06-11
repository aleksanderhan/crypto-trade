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
            'n_steps': params['n_steps'],
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'ent_coef': params['ent_coef'],
            'clip_range': params['cliprange'],
            'clip_range_vf': params['clip_range_vf']
        }
    except:
        model_params = {
            'n_steps': 476,
            'gamma': 0.90606675817696,
            'learning_rate': 0.025412921449645007,
            'ent_coef': 0.0008681344845273617,
            'clip_range': 0.2025550612826041,
            'clip_range_vf': 0.28148595837701307
        }

    return model_params