import gym
from gym import spaces
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pmdarima as pm
import requests
import json
import random
import warnings
from collections import deque
from empyrical import sortino_ratio, calmar_ratio, omega_ratio
from time import perf_counter

from visualize import TradingGraph

warnings.filterwarnings("ignore")


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
                confidence_interval,
                use_forecast,
                initial_step=100,
                arima_order=(0, 1, 0),
                fee=0.005):
        
        super(CryptoTradingEnv, self).__init__()

        self.df = df.fillna(method='bfill')
        self.visualization = None
        self.coins = coins
        self.max_steps = len(df.index) - 1
        self.fee = fee
        
        self.max_initial_balance = max_initial_balance
        self.initial_balance = random.randint(1000, max_initial_balance)
        self.initial_step = initial_step
        self.current_step = initial_step
        self.max_net_worth = self.initial_balance
        
        self.portfolio = {}
        self.balance = deque(maxlen=2)
        self.net_worth = []
        self.trades = []

        # Reward
        self.reward_func = reward_func
        self.reward_len = reward_len

        # Forecast
        self.forecast_len = forecast_len
        self.confidence_interval = confidence_interval
        self.use_forecast = use_forecast
        self.arima_order = arima_order
        self.arima_models = {}

        # Buy/sell/hold for each coin
        self.action_space = spaces.Box(low=np.array([-1, -1, -1], dtype=np.float32), high=np.array([1, 1, 1], dtype=np.float32), dtype=np.float32)
        
        # (num_coins * (portefolio value & candles & 3*(forecast_len-1)) + (balance & net worth & timestamp))
        observation_space_len = (len(coins) * (6 + 3*(self.forecast_len - 1)) + 3) if self.use_forecast else len(coins) * 6 + 3
        self.observation_space = spaces.Box(
            low=-MAX_VALUE,
            high=MAX_VALUE, 
            shape=(observation_space_len,),
            dtype=np.float32
        )

        self.reset()


    def step(self, action):
        t0 = perf_counter()
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1

        #delay_modifier = (self.current_step / self.max_steps)
        #profit = self.get_profit()

        obs = self._next_observation()
        reward = self._get_reward(self.reward_func)

        lost_90_percent_net_worth = float(self.net_worth[-1]) < (self.initial_balance / 10)
        done = lost_90_percent_net_worth or self.current_step >= self.max_steps

        info = {
            'current_step': self.current_step, 
            'last_trade': self._get_last_trade(),
            'profit': self.get_profit(),
            'max_steps': self.max_steps
        }
        t1 = perf_counter()
        #print('step dt', t1-t0)
        return obs, reward, done, info


    def reset(self):
        # Set the current step to a random point within the data frame
        self.current_step = self.initial_step
        self.initial_balance = random.randint(1000, self.max_initial_balance)
        self.max_net_worth = self.initial_balance
        self.trades = []

        for coin in self.coins:
            self.portfolio[coin] = deque(maxlen=2)
            for i in range(2):
                self.portfolio[coin].append(0)

        for i in range(self.initial_step):
            self.balance.append(self.initial_balance)
            self.net_worth.append(self.initial_balance)

        for coin in self.coins:
            past_close_values = self.df.loc[0 : self.initial_step - 1, coin + '_close'].values
            arima = pm.auto_arima(past_close_values, 
                                start_p=1, start_q=1, d=0, max_p=5, max_q=5, 
                                suppress_warnings=True, stepwise=True, error_action='ignore')

            self.arima_models[coin] = arima

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
            reward = np.mean(returns)
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
        frame = []
        for coin in self.coins:
            # Price data
            open_values = self.df.loc[self.current_step - 1: self.current_step, coin + '_open'].values
            high_values = self.df.loc[self.current_step - 1: self.current_step, coin + '_high'].values
            low_values = self.df.loc[self.current_step - 1: self.current_step, coin + '_low'].values
            close_values = self.df.loc[self.current_step - 1: self.current_step, coin + '_close'].values
            volume_values = self.df.loc[self.current_step - 1: self.current_step, coin + '_volume'].values
            #print(f'open_values {coin}', len(open_values))
            #print(f'high_values {coin}', len(high_values))
            #print(f'low_values {coin}', len(low_values))
            #print(f'close_values {coin}', len(close_values))
            #print(f'volume_values {coin}', len(volume_values))
            frame.append(np.diff(np.log(open_values)))
            frame.append(np.diff(np.log(high_values)))
            frame.append(np.diff(np.log(low_values)))
            frame.append(np.diff(np.log(close_values)))
            frame.append(np.diff(np.log(volume_values)))

            # Portefolio
            frame.append(np.diff(np.log(np.array(self.portfolio[coin]) + 1))) # +1 dealing with 0 log
            #print(f'portefolio {coin}', len(self.portfolio[coin]))
            if self.use_forecast:
                # Forecast prediction
                forecast, conf_int = self._get_forecast(coin)
                frame.append(np.diff(np.log(forecast)))
                #print(f'forecast {coin}', len(forecast))

                # Forecast confidence interval
                ci_start = conf_int.flatten()[0::2]
                ci_end = conf_int.flatten()[1::2]
                frame.append(np.diff(np.log(ci_start)))
                frame.append(np.diff(np.log(ci_end)))
                #print(f'ci_start {coin}', len(ci_start))
                #print(f'ci_end {coin}', len(ci_end))

        # Time
        timestamp_values = self.df.loc[self.current_step - 1: self.current_step, 'timestamp'].values
        frame.append(np.diff(np.log(timestamp_values)))
        #print('timestamp_values', len(timestamp_values))

        # Net worth and balance
        frame.append(np.diff(np.log(np.array(self.balance) + 1))) # +1 dealing with 0 log
        frame.append(np.diff(np.log(self.net_worth[self.current_step-1:self.current_step+1])))
        #print('balance', len(self.balance))
        #print('net_worth', len(self.net_worth[self.current_step-1:self.current_step+1]))

        t1 = perf_counter()
        print('obs_dt', t1-t0)
        obs = np.nan_to_num(np.concatenate(frame), posinf=MAX_VALUE, neginf=-MAX_VALUE)
        #print(obs)
        #print('len obs', len(obs))
        #print()
        return obs


    def _get_forecast(self, coin):
        new_observation = self.df.at[self.current_step, coin + '_close']
        
        self.arima_models[coin].update(new_observation)
        forecast, conf_int = self.arima_models[coin].predict(n_periods=self.forecast_len, 
                                                            return_conf_int=True, 
                                                            alpha=(1 - self.confidence_interval))

        return forecast, conf_int


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