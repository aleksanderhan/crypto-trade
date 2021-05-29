import gym
from gym import spaces
import pandas as pd
import numpy as np
import requests
import json
import random
from collections import deque

from visualize import TradingGraph

LOOKBACK_WINDOW_SIZE = 50
MAX_VALUE = 3.4e38 # Max float 32


class CryptoTradingEnv(gym.Env):
    """A crypto trading environment for OpenAI gym"""
    metadata = {'render.modes': ['console', 'human']}

    def __init__(self, frame_size, max_initial_balance, df, coins, fee=0.005, debug=False):
        super(CryptoTradingEnv, self).__init__()
        self.frame_size = frame_size
        self.max_initial_balance = max_initial_balance
        self.initial_balance = random.randint(1000, max_initial_balance)
        self.df = df
        self.coins = coins
        self.max_steps = len(df.index) - frame_size
        self.fee = fee
        self.visualization = None
        self.current_step = frame_size
        self.portfolio = {}
        self.max_net_worth = self.initial_balance
        self.balance = deque(maxlen=frame_size)
        self.net_worth = deque(maxlen=frame_size)
        self.training = True
        self.trades = []
        self.debug = debug

        # Buy/sell/hold for each coin
        self.action_space = spaces.Box(low=np.array([-1, -1, -1], dtype=np.float16), high=np.array([1, 1, 1], dtype=np.float16), dtype=np.float16)

        # prices over the last few days and portfolio status
        self.observation_space = spaces.Box(
            low=-MAX_VALUE,
            high=MAX_VALUE, 
            shape=((len(coins) * 6 + 3) * (frame_size-1),), # (num_coins * (portefolio value & candles) + (balance & net worth & timestamp)) * frame_size
            dtype=np.float32
        )


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1

        #delay_modifier = (self.current_step / self.max_steps)
        #profit = self.net_worth[-1] - self.initial_balance

        obs = self._next_observation()
        reward = self.net_worth[-1] - self.net_worth[-2]
        done = self.net_worth[-1] <= 0 or self.current_step >= self.max_steps -1 

        return obs, reward, done, {'current_step': self.current_step, 'last_trade': self._get_last_trade()}


    def reset(self, training=True):
        # Set the current step to a random point within the data frame
        self.current_step = self.frame_size
        self.initial_balance = random.randint(1000, self.max_initial_balance)
        self.max_net_worth = self.initial_balance
        self.trades = []

        for coin in self.coins:
            self.portfolio[coin] = deque(maxlen=self.frame_size)
            for _ in range(self.frame_size):
                self.portfolio[coin].append(0)

        for _ in range(self.frame_size):
            self.balance.append(self.initial_balance)
            self.net_worth.append(self.initial_balance)

        return self._next_observation()


    


    def _take_action(self, action):
        action_type = action[0]
        amount = 0.5*(action[1] - 1) + 1 # https://tiagoolivoto.github.io/metan/reference/resca.html
        
        coin_action = ((len(self.coins) - 1) / 2) * (action[2] - 1) + len(self.coins) - 1
        coin = self.coins[int(coin_action)]

        # Set the current price to a random price within the time step
        current_price = random.uniform(self.df.loc[self.current_step, coin + '_low'], self.df.loc[self.current_step, coin + '_high'])

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
                'type': 'buy'
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
                'type': 'sell'
                })

        self.net_worth.append(self._calculate_net_worth())
        if self.net_worth[-1] > self.max_net_worth:
            self.max_net_worth = self.net_worth[-1]


    def _next_observation(self):
        frame = np.array([])
        for coin in self.coins:
            open_values = self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_open'].values
            high_values = self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_high'].values
            low_values = self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_low'].values
            close_values = self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_close'].values
            volume_values = self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_volume'].values

            frame = np.concatenate((frame, np.array([open_values[i] - open_values[i-1] for i in range(1, len(open_values))])))
            frame = np.concatenate((frame, np.array([high_values[i]- high_values[i-1] for i in range(1, len(high_values))])))
            frame = np.concatenate((frame, np.array([low_values[i] - low_values[i-1] for i in range(1, len(low_values))])))
            frame = np.concatenate((frame, np.array([close_values[i] - close_values[i-1] for i in range(1, len(close_values))])))
            frame = np.concatenate((frame, np.array([volume_values[i] - volume_values[i-1] for i in range(1, len(volume_values))])))

            frame = np.concatenate((frame, np.array([self.portfolio[coin][i] - self.portfolio[coin][i-1] for i in range(1, len(self.portfolio[coin]))])))

        timestamp_values = self.df.loc[self.current_step - self.frame_size +1: self.current_step, 'timestamp'].values
        frame = np.concatenate((frame, np.array([timestamp_values[i] - timestamp_values[i-1] for i in range(1, len(timestamp_values))])))

        frame = np.concatenate((frame, np.array([self.balance[i] - self.balance[i-1] for i in range(1, len(self.balance))])))
        frame = np.concatenate((frame, np.array([self.net_worth[i] - self.net_worth[i-1] for i in range(1, len(self.net_worth))])))
        return frame
        

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


    def render(self, mode='console', title=None, **kwargs):
        # Render the environment to the screen
        profit = self.net_worth[-1] - self.initial_balance

        if mode == 'console':
            print(f'Step: {self.current_step} of {self.max_steps}')
            print(f'Balance: {self.balance[-1]}')
            print(f'Net worth: {self.net_worth[-1]} (Max net worth: {self.max_net_worth})')
            print(f'Profit: {profit}')
            print(f'Last_trade: {self._get_last_trade()}')

        elif mode == 'human':
            if self.visualization == None:
                self.visualization = TradingGraph(self.df, self.coins, title)
            
            if self.current_step > LOOKBACK_WINDOW_SIZE:        
                self.visualization.render(self.current_step, self.net_worth[-1], self.trades, window_size=LOOKBACK_WINDOW_SIZE)
        
        return profit


    def close(self):
        if self.visualization is not None:
            self.visualization.close()
            self.visualization = None