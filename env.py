import gym
from gym import spaces
import pandas as pd
import numpy as np
import requests
import json
import random
from collections import deque


class CryptoTradingEnv(gym.Env):
    """A crypto trading environment for OpenAI gym"""
    metadata = {'render.modes': ['console']}

    def __init__(self, frame_size, initial_balance, df, coins, max_steps):
        super(CryptoTradingEnv, self).__init__()
        self.frame_size = frame_size
        self.initial_balance = initial_balance
        self.df = df
        self.coins = coins
        self.max_steps = max_steps

        # Buy/sell/hold for each coin
        self.action_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([3, 1, len(coins)-1]), dtype=np.float16)

        # prices over the last few days and portfolio status
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(14, 5), dtype=np.float16)


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1

        delay_modifier = (self.current_step / self.max_steps)

        obs = self._next_observation()
        reward = self.net_worth[-1] * delay_modifier
        done = self.net_worth[-1] <= 0 or self.current_step >= self.max_steps

        return obs, reward, done, {'current_step': self.current_step}


    def reset(self):
        self.current_step = self.frame_size

        self.portfolio = {}
        for coin in self.coins:
            self.portfolio[coin] = deque(maxlen=self.frame_size)
            for _ in range(self.frame_size):
                self.portfolio[coin].append(0)


        self.balance = deque(maxlen=self.frame_size)
        self.net_worth = deque(maxlen=self.frame_size)
        for _ in range(self.frame_size):
            self.balance.append(self.initial_balance)
            self.net_worth.append(self.initial_balance)
        self.max_net_worth = self.initial_balance

        obs = self._next_observation()
        return obs


    def _take_action(self, action):
        action_type = int(action[0])
        amount = action[1]
        coin = self.coins[int(action[2])]

        # Set the current price to a random price within the time step
        current_price = random.uniform(self.df.loc[self.current_step, coin + "_open"], self.df.loc[self.current_step, coin + "_close"])

        if action_type == 0:
            # Buy amount % of balance in shares
            total_possible = self.balance[-1] / current_price
            coins_bought = total_possible * amount
            self.balance.append(self.balance[-1] - coins_bought * current_price)
            self.portfolio[coin].append(self.portfolio[coin][-1] + coins_bought)

        elif action_type == 1:
            # Sell amount % of shares held
            coins_sold = self.portfolio[coin][-1] * amount
            self.balance.append(self.balance[-1] + coins_sold * current_price)
            self.portfolio[coin].append(self.portfolio[coin][-1] - coins_sold)


        self.net_worth.append(self._calculate_net_worth())
        if self.net_worth[-1] > self.max_net_worth:
            self.max_net_worth = self.net_worth[-1]


    def _next_observation(self):
        frame = []
        for coin in self.coins:
            frame.append(self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_open'].values)
            frame.append(self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_high'].values)
            frame.append(self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_low'].values)
            frame.append(self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_close'].values)
            frame.append(self.df.loc[self.current_step - self.frame_size +1: self.current_step, coin + '_volume'].values)
            frame.append(np.array(self.portfolio[coin]))

        frame.append(np.array(self.balance))
        frame.append(np.array(self.net_worth))

        return np.array(frame)


    def _calculate_net_worth(self):
        portfolio_value = 0
        for coin in self.coins:
            coin_price = (self.df.at[self.current_step, coin + '_low'] + self.df.at[self.current_step, coin + '_high'])/2
            portfolio_value += self.portfolio[coin][-1] * coin_price

        return self.balance[-1] + portfolio_value


    def render(self, mode='console', close=False):
        # Render the environment to the screen
        profit = self.net_worth[-1] - self.initial_balance

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance[-1]}')
        print(f'Net worth: {self.net_worth[-1]} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')

