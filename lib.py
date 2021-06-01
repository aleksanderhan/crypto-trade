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
    try:
        study = optuna.load_study(study_name='optimize_profit', storage='sqlite:///params.db')

        params = study.best_trial.params
        print(params)

        env_params = {
            'reward_func': params['reward_func'],
            'reward_len': int(params['reward_len']),
            'forecast_len': int(params['forecast_len']),
            'lookback_interval': int(params['lookback_interval']),
            'confidence_interval': params['confidence_interval']
        }
        model_params = {
            'n_steps': int(params['n_steps']),
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'ent_coef': params['ent_coef'],
            'cliprange': params['clip_range'],
            'lam': params['lam']
        }
    except:
        env_params = {
            'reward_func': 'simple',
            'reward_len': 10,
            'forecast_len': 10,
            'lookback_interval': 10,
            'confidence_interval': 0.8
        }
        model_params = {
            'n_steps': 215,
            'gamma': 0.9287084025015536,
            'learning_rate': 0.014929699035787503,
            'ent_coef': 0.00043960436532134166,
            'cliprange': 0.12840973632198896,
            'lam': 0.95
        }

    return env_params, model_params