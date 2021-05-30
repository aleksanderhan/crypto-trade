import requests
import optuna
import pandas as pd


DEFAULT_FRAME_SIZE = 100



def get_data(start_time, end_time, coins, granularity):
    coinsStr = ','.join(coins)
    r = requests.get(f'http://127.0.0.1:5000/data?start_time={start_time}&end_time={end_time}&coins={coinsStr}&granularity={granularity}')

    df = pd.DataFrame.from_dict(r.json())
    print(df)
    df.index = df.index.astype(int)
    return df


def load_params():
    study = optuna.load_study(study_name='optimize_profit', storage='sqlite:///params.db')

    try:
        params = study.best_trial.params
        print(params)
        env_params = {'frame_size': params['frame_size']}
        model_params = {
            'n_steps': int(params['n_steps']),
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'ent_coef': params['ent_coef'],
            'clip_range': params['clip_range'],
            'clip_range_vf': params['clip_range_vf']
        }

        return env_params, model_params
    except ValueError:
        return {'frame_size': DEFAULT_FRAME_SIZE}, {}