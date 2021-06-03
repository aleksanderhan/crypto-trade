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
            'reward_len': params['reward_len'],
            'forecast_len': params['forecast_len'],
            'lookback_interval': params['lookback_interval'],
            'confidence_interval': params['confidence_interval']
        }
        model_params = {
            'policy': params['policy'],
            'n_steps': params['n_steps'],
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'ent_coef': params['ent_coef'],
            'cliprange': params['clip_range'],
            'lam': params['lam']
        }
    except:
        env_params = {
            'reward_func': 'sortino',
            'reward_len': 4,
            'forecast_len': 8,
            'lookback_interval': 50,
            'confidence_interval': 0.738
        }
        model_params = {
            'policy': 'MlpPolicy',
            'n_steps': 1849,
            'gamma': 0.988,
            'learning_rate': 0.0028,
            'ent_coef': 0.0001738,
            'cliprange': 0.24178,
            'lam': 0.933
        }

    return env_params, model_params