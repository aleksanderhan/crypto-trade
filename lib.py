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
            'forecast_len': params['forecast_len'],
            'lookback_interval': params['lookback_interval'],
            'confidence_interval': params['confidence_interval'],
            'arima_order': (params['arima_p'], params['arima_p'], params['arima_q']),
            'use_forecast': params['use_forecast']
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
            'reward_len': 3,
            'forecast_len': 1,
            'lookback_interval': 32,
            'confidence_interval': 0.8460604699339368,
            'arima_order': (1, 0, 1),
            'use_forecast': True
        }
        model_params = {
            'n_steps': 1212,
            'gamma': 0.9875622908741607,
            'learning_rate': 0.11727616345950494,
            'ent_coef': 2.5017991913510806e-07,
            'cliprange': 0.3193524886168006,
            'lam': 0.9940652382030966
        }

    return env_params, model_params