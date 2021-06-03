import gym
from gym import spaces
import pandas as pd
import numpy as np
import requests
import json
import random
import warnings
from collections import deque
from empyrical import sortino_ratio, calmar_ratio, omega_ratio
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from time import perf_counter

from visualize import TradingGraph

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', UserWarning)


LOOKBACK_WINDOW_SIZE = 100
MAX_VALUE = 3.4e38 # ~Max float 32


class CryptoTradingEnv(gym.Env):
    """A crypto trading environment for OpenAI gym"""
    metadata = {'render.modes': ['console', 'human']}

    def __init__(self, 
                df, 
                coins, 
                max_initial_balance, 
                reward_func,
                reward_len,
                forecast_len,
                lookback_interval,
                confidence_interval,
                arima_order=(0, 1, 0),
                fee=0.005):
        
        super(CryptoTradingEnv, self).__init__()

        self.max_initial_balance = max_initial_balance
        self.initial_balance = random.randint(1000, max_initial_balance)
        self.df = df.fillna(method='bfill')
        self.coins = coins
        self.max_steps = len(df.index) - 1
        self.fee = fee
        self.visualization = None
        self.current_step = 1
        self.portfolio = {}
        self.max_net_worth = self.initial_balance
        self.balance = deque(maxlen=2)
        self.net_worth = []
        self.trades = []
        self.reward_func = reward_func
        self.reward_len = reward_len
        self.forecast_len = forecast_len
        self.lookback_interval = lookback_interval
        self.confidence_interval = confidence_interval
        self.arima_order = arima_order

        # Buy/sell/hold for each coin
        self.action_space = spaces.Box(low=np.array([-1, -1, -1], dtype=np.float16), high=np.array([1, 1, 1], dtype=np.float32), dtype=np.float32)

        # prices over the last few days and portfolio status
        self.observation_space = spaces.Box(
            low=-MAX_VALUE,
            high=MAX_VALUE, 
            shape=((len(coins) * (6 + self.forecast_len + 2*(self.forecast_len - 2)) + 3 + 2),), # (num_coins * (portefolio value & candles & forecast_len*3) + (balance & net worth & timestamp))
            dtype=np.float32
        )


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1

        #delay_modifier = (self.current_step / self.max_steps)
        #profit = self.get_profit()

        obs = self._next_observation()
        reward = self._get_reward(self.reward_func)
        done = self.net_worth[-1] <= 0 or self.current_step >= self.max_steps

        return obs, reward, done, {
            'current_step': self.current_step, 
            'last_trade': self._get_last_trade(),
            'profit': self.get_profit(),
            'max_steps': self.max_steps
            }


    def reset(self):
        # Set the current step to a random point within the data frame
        self.current_step = 1
        self.initial_balance = random.randint(1000, self.max_initial_balance)
        self.max_net_worth = self.initial_balance
        self.trades = []

        for coin in self.coins:
            self.portfolio[coin] = deque(maxlen=2)
            for i in range(2):
                self.portfolio[coin].append(0)

        for i in range(2):
            self.balance.append(self.initial_balance)
            self.net_worth.append(self.initial_balance)

        return self._next_observation()


    def _get_reward(self, reward_func=None):        
        returns = np.diff(self.net_worth[-self.reward_len:])

        if self.reward_func == 'sortino':
            reward = sortino_ratio(returns)
        elif self.reward_func == 'calmar':
            reward = calmar_ratio(returns)
        elif self.reward_func == 'omega':
            reward = omega_ratio(returns)
        elif self.reward_func == 'simple':
            reward = returns[-1]
        else:
            raise NotImplementedError

        return reward if not np.isinf(reward) and not np.isnan(reward) else 0


    def _take_action(self, action):
        action = np.nan_to_num(action, posinf=1, neginf=-1)
        action_type = action[0]
        amount = 0.5*(action[1] - 1) + 1 # https://tiagoolivoto.github.io/metan/reference/resca.html

        coin_action = ((len(self.coins) - 1) / 2) * (action[2] - 1) + len(self.coins) - 1
        coin = self.coins[int(coin_action)]

        # Set the current price to a random price within the time step
        current_price = random.uniform(self.df.loc[self.current_step, coin + '_low'], self.df.loc[self.current_step, coin + '_high'])

        if current_price > 0: # Price is 0 before ICO
            if action_type  <= 1 and action_type > 1/3:
                # Buy amount % of balance in shares
                total_possible = self.balance[-1] / current_price
                coins_bought = max(0, total_possible * (1 - self.fee) * amount)
                cost = coins_bought * current_price
        
                self.balance.append(self.balance[-1] - cost)
                self.portfolio[coin].append(self.portfolio[coin][-1] + coins_bought)

                if coins_bought > 0:
                  self.trades.append({
                    'step': self.current_step,
                    'coin': coin,
                    'coins_bought': coins_bought, 
                    'total': cost,
                    'type': 'buy',
                    'price': current_price
                    })

            elif action_type >= -1 and action_type < -1/3:
                # Sell amount % of shares held
                coins_sold = max(0, self.portfolio[coin][-1] * amount)

                self.balance.append(self.balance[-1] + coins_sold * (1 - self.fee) * current_price)
                self.portfolio[coin].append(self.portfolio[coin][-1] - coins_sold)

                if coins_sold > 0:
                  self.trades.append({
                    'step': self.current_step,
                    'coin': coin,
                    'coins_sold': coins_sold, 
                    'total': coins_sold * current_price,
                    'type': 'sell',
                    'price': current_price
                    })
            else:
                # Hold
                pass


        self.net_worth.append(self._calculate_net_worth())
        if self.net_worth[-1] > self.max_net_worth:
            self.max_net_worth = self.net_worth[-1]


    def _next_observation(self):
        t0 = perf_counter()
        frame = np.array([])
        for coin in self.coins:
            # Price data
            open_values = self.df.loc[self.current_step - 1: self.current_step, coin + '_open'].values
            high_values = self.df.loc[self.current_step - 1: self.current_step, coin + '_high'].values
            low_values = self.df.loc[self.current_step - 1: self.current_step, coin + '_low'].values
            close_values = self.df.loc[self.current_step - 1: self.current_step, coin + '_close'].values
            volume_values = self.df.loc[self.current_step - 1: self.current_step, coin + '_volume'].values
            frame = np.concatenate((frame, np.diff(np.log(open_values))))
            frame = np.concatenate((frame, np.diff(np.log(high_values))))
            frame = np.concatenate((frame, np.diff(np.log(low_values))))
            frame = np.concatenate((frame, np.diff(np.log(close_values))))
            frame = np.concatenate((frame, np.diff(np.log(volume_values))))

            # Portefolio
            frame = np.concatenate((frame, np.diff(np.log(np.array(self.portfolio[coin]) + 1)))) # +1 dealing with 0 log

            # Forecast prediction
            forecast = self._get_forecast(coin)
            frame = np.concatenate((frame, np.diff(np.log(forecast.predicted_mean))))

            # Forecast confidence interval
            ci_start = forecast.conf_int().flatten()[0::2]
            ci_end = forecast.conf_int().flatten()[1::2]
            frame = np.concatenate((frame, np.diff(np.log(ci_start))))
            frame = np.concatenate((frame, np.diff(np.log(ci_end))))

        # Time
        timestamp_values = self.df.loc[self.current_step - 1: self.current_step, 'timestamp'].values
        frame = np.concatenate((frame, np.diff(np.log(timestamp_values))))

        # Net worth and balance
        frame = np.concatenate((frame, np.diff(np.log(np.array(self.balance) + 1)))) # +1 dealing with 0 log
        frame = np.concatenate((frame, np.diff(np.log(self.net_worth[self.current_step-1:self.current_step+1]))))
        t1 = perf_counter()
        #print('obs_dt', t1-t0)

        return np.nan_to_num(frame, posinf=MAX_VALUE, neginf=-MAX_VALUE)


    def _get_forecast(self, coin):
        past_close_values = self.df.loc[self.current_step - self.lookback_interval: self.current_step, coin + '_close'].values

        if len(past_close_values) < 3: # Padding values at first frame
            past_close_values = np.insert(past_close_values, 0, past_close_values[0])

        forecast_model = ARIMA(past_close_values,
            order=self.arima_order,
            enforce_stationarity=False,
            enforce_invertibility=False)

        model_fit = forecast_model.fit(start_params=[0, 0, 0, 0, 1, 1])
        forecast = model_fit.get_forecast(steps=self.forecast_len, alpha=(1 - self.confidence_interval), typ='levels')

        return forecast


    def _calculate_net_worth(self):
        portfolio_value = 0
        for coin in self.coins:
            coin_price = (self.df.at[self.current_step, coin + '_low'] + self.df.at[self.current_step, coin + '_high'])/2
            portfolio_value += self.portfolio[coin][-1] * coin_price
        return self.balance[-1] + portfolio_value


    def _get_last_trade(self):
        try:
            return self.trades[-1]
        except IndexError:
            return None


    def get_profit(self):
        return self.net_worth[-1] - self.initial_balance


    def render(self, mode='console', title=None, **kwargs):
        # Render the environment to the screen
        profit = self.get_profit()        
        
        if mode == 'console':
            for coin in self.coins:
                print(coin, self.portfolio[coin][-1])

            print(f'Last_trade: {self._get_last_trade()}')
            print(f'Step: {self.current_step} of {self.max_steps}')
            print(f'Balance: {self.balance[-1]} (Initial balance: {self.initial_balance})')
            print(f'Net worth: {self.net_worth[-1]} (Max net worth: {self.max_net_worth})')
            print(f'Profit: {profit}')
            print(f'Reward: {self._get_reward()}')
            print()

        elif mode == 'human':
            if self.visualization == None:
                self.visualization = TradingGraph(self.df, self.coins, title)
            
            if self.current_step > LOOKBACK_WINDOW_SIZE:        
                self.visualization.render(self.current_step, self.net_worth[-1], self.trades, window_size=LOOKBACK_WINDOW_SIZE)

            print(f'Step: {self.current_step} of {self.max_steps}')
            print(f'Balance: {self.balance[-1]} (Initial balance: {self.initial_balance})')
            print(f'Net worth: {self.net_worth[-1]} (Max net worth: {self.max_net_worth})')
            print(f'Profit: {profit}')
            print()


    def close(self):
        if self.visualization is not None:
            self.visualization.close()
            self.visualization = None