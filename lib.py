import requests
import optuna
import pandas as pd




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
        env_params = {'frame_size': int(params['frame_size'])}
        model_params = {
            'n_steps': int(params['n_steps']),
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'ent_coef': params['ent_coef'],
            'clip_range': params['clip_range'],
            'clip_range_vf': params['clip_range_vf']
        }
    except ValueError:
        env_params = {'frame_size': 25}, {}
    	model_params = {
            'n_steps': 215,
            'gamma': 0.9287084025015536,
            'learning_rate': 0.014929699035787503,
            'ent_coef': 0.00043960436532134166,
            'clip_range': 0.12840973632198896,
            'clip_range_vf': 0.34228823879560166
        }

    return env_params, model_params