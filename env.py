import gym
from gym import spaces
import pandas as pd
import numpy as np
import requests
import json
import random
from collections import deque

from visualize import StockTradingGraph

LOOKBACK_WINDOW_SIZE = 50


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
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float16)

        # prices over the last few days and portfolio status
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(
                len(coins) * 6 + 3, # num_coins * (portefolio value & candles) + (balance & net worth & timestamp)
                frame_size
            ), 
            dtype=np.float32
        )


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1

        delay_modifier = (self.current_step / self.max_steps)

        '''
        print('net worth', self.net_worth[-1])
        print('maxstep', self.max_steps)
        print('current_step', self.current_step)
        print(self.portfolio)
        print()
        '''

        obs = self._next_observation()
        reward = self.net_worth[-1] * delay_modifier
        done = self.net_worth[-1] <= 0 or self.current_step >= self.max_steps -1 

        return obs, reward, done, {'current_step': self.current_step, 'trades': self.trades}


    def reset(self, training=True):
        # Set the current step to a random point within the data frame
        self.current_step = self.frame_size - 1
        self.initial_balance = random.randint(1000, self.max_initial_balance)
        self.trades = []

        for coin in self.coins:
            self.portfolio[coin] = deque(maxlen=self.frame_size)
            for _ in range(self.frame_size):
                self.portfolio[coin].append(0)

        for _ in range(self.frame_size):
            self.balance.append(self.initial_balance)
            self.net_worth.append(self.initial_balance)

        return self._next_observation()


    def render(self, mode='console', title=None, **kwargs):
        # Render the environment to the screen
        profit = self.net_worth[-1] - self.initial_balance
        if mode == 'console':
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance[-1]}')
            print(f'Net worth: {self.net_worth[-1]} (Max net worth: {self.max_net_worth})')
            print(f'Profit: {profit}')
            print(f'Portfolio: {self.portfolio}')
        elif mode == 'human':
            if self.visualization == None:
              self.visualization = StockTradingGraph(self.df, title)
            
            if self.current_step > LOOKBACK_WINDOW_SIZE:        
              self.visualization.render(self.current_step, self.net_worth[-1], self.trades, window_size=LOOKBACK_WINDOW_SIZE)
        
        return profit


    def _take_action(self, action):
        action_type = action[0]
        amount = 0.5*(action[1] - 1) + 1 # https://tiagoolivoto.github.io/metan/reference/resca.html
        
        coin_action = ((len(self.coins) - 1) / 2) * (action[2] - 1) + len(self.coins) - 1


        coin = self.coins[int(coin_action)]
        
        if self.debug:
            print()
            print('action_type', action_type)
            print('amount', amount)
            print('coin_action', coin_action)
            print('coin', coin)
            
        # Set the current price to a random price within the time step
        current_price = random.uniform(self.df.loc[self.current_step, coin + '_low'], self.df.loc[self.current_step, coin + '_high'])

        if self.debug:
            print('current_price', current_price)
            print('balance', self.balance)
            print('portfolio', self.portfolio)

        if action_type  <= 1 and action_type > 1/3:
            # Buy amount % of balance in shares
            total_possible = self.balance[-1] / current_price
            coins_bought = max(0, total_possible * (1 - self.fee) * amount)

            if self.debug:
                print()
                print(total_possible)
                print(self.fee)
                print(1-self.fee)
                print(amount)
                print()

            cost = coins_bought * current_price
            
            if self.debug:
                print('total_possible', total_possible)
                print('coins_bought', coins_bought)

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
            if self.debug:
                print('coins_sold', coins_sold)
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
        frame = []
        for coin in self.coins:
            #print(self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_open'].values)
            frame.append(self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_open'].values)
            frame.append(self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_high'].values)
            frame.append(self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_low'].values)
            frame.append(self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_close'].values)
            frame.append(self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_volume'].values)
            frame.append(np.array(self.portfolio[coin]))

        frame.append(self.df.loc[self.current_step - self.frame_size +1: self.current_step, 'timestamp'].values)
        frame.append(np.array(self.balance))
        frame.append(np.array(self.net_worth))

        return np.array(frame)


    def _calculate_net_worth(self):
        portfolio_value = 0
        for coin in self.coins:
            coin_price = (self.df.at[self.current_step, coin + '_low'] + self.df.at[self.current_step, coin + '_high'])/2
            portfolio_value += self.portfolio[coin][-1] * coin_price
        return self.balance[-1] + portfolio_value

