import gym
from gym import spaces
import pandas as pd
import numpy as np
import requests
import json
import random
from collections import deque
from empyrical import sortino_ratio, calmar_ratio, omega_ratio

from visualize import TradingGraph

LOOKBACK_WINDOW_SIZE = 100
MAX_VALUE = np.inf #3.4e38 # Max float 32


class CryptoTradingEnv(gym.Env):
    """A crypto trading environment for OpenAI gym"""
    metadata = {'render.modes': ['console', 'human']}

    def __init__(self, max_initial_balance, df, coins, reward_func, frame_size=50, fee=0.005, debug=False):
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
        self.reward_func = reward_func

        # Buy/sell/hold for each coin
        self.action_space = spaces.Box(low=np.array([-1, -1, -1], dtype=np.float16), high=np.array([1, 1, 1], dtype=np.float16), dtype=np.float16)

        # prices over the last few days and portfolio status
        self.observation_space = spaces.Box(
            low=-MAX_VALUE,
            high=MAX_VALUE, 
            shape=((len(coins) * 6 + 3) * (frame_size-1),), # (num_coins * (portefolio value & candles) + (balance & net worth & timestamp)) * frame_size -1
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


    def _get_reward(self, reward_func=None):        
        returns = np.diff(self.net_worth)

        if reward_func == 'sortino':
            reward = sortino_ratio(returns)
        elif reward_func == 'calmar':
            reward = calmar_ratio(returns)
        elif reward_func == 'omega':
            reward = omega_ratio(returns)
        else:
            reward = np.mean(returns)

        return reward if not np.isinf(reward) and not np.isnan(reward) else 0


    def _take_action(self, action):
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
        frame = np.array([])
        for coin in self.coins:
            open_values = self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_open'].values
            high_values = self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_high'].values
            low_values = self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_low'].values
            close_values = self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_close'].values
            volume_values = self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_volume'].values

            frame = np.concatenate((frame, np.diff(open_values)))
            frame = np.concatenate((frame, np.diff(high_values)))
            frame = np.concatenate((frame, np.diff(low_values)))
            frame = np.concatenate((frame, np.diff(close_values)))
            frame = np.concatenate((frame, np.diff(volume_values)))

            frame = np.concatenate((frame, np.diff(self.portfolio[coin])))

        timestamp_values = self.df.loc[self.current_step - self.frame_size +1: self.current_step, 'timestamp'].values
        frame = np.concatenate((frame, np.diff(timestamp_values)))

        frame = np.concatenate((frame, np.diff(self.balance)))
        frame = np.concatenate((frame, np.diff(self.net_worth)))
        return np.nan_to_num(frame)


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
            print(f'Last_trade: {self._get_last_trade()}')
            for coin in self.coins:
                print(coin, self.portfolio[coin][-1])
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