import gym
from gym import spaces
import pandas as pd
import numpy as np
from numpy import array, log, diff, nan_to_num, concatenate
import matplotlib.pylab as plt
import requests
import json
import random
import warnings
from collections import deque
from time import perf_counter
from empyrical import sortino_ratio
from ta import add_all_ta_features

from visualize import TradingGraph

warnings.filterwarnings("ignore")


LOOKBACK_WINDOW_SIZE = 100
MAX_VALUE = 3.4e38 # ~Max float32


class CryptoTradingEnv(gym.Env):
    """A crypto trading environment for OpenAI gym"""
    metadata = {'render.modes': ['console', 'human']}
    version = '1.3'

    def __init__(self, 
                df, 
                coins,
                initial_balance,
                lookback_len,
                fee=0.0):
        
        super(CryptoTradingEnv, self).__init__()

        self.visualization = None
        self.df = df.fillna(method='bfill')
        self.coins = coins
        self.fee = fee
        self.max_steps = len(df.index) - 1
        self.lookback_len = lookback_len
        self.current_step = lookback_len

        self.initial_balance = initial_balance
        self.max_net_worth = initial_balance
        
        self.balance = deque(maxlen=lookback_len)
        self.real_net_worth = deque(maxlen=lookback_len)
        self.hodl_net_worth = deque(maxlen=lookback_len)
        self.portfolio = {}
        self.start_portfolio = {}
        self.portfolio_value = {}

        self.max_postitions = 5
        self.positions_held_now = 0
        self.positions_held = deque(maxlen=lookback_len)
        self.trades = []
        self.fees_payed = 0

        self.idiot_moves = 0
        self.last_reward = 0
        self.cumulative_reward = 0

        # Augument dataframe with technical analysis features
        for coin in coins:
            self.df = add_all_ta_features(self.df, 
                open=coin+'_open', 
                high=coin+'_high', 
                low=coin+'_low', 
                close=coin+'_close', 
                volume=coin+'_volume',
                colprefix=coin+'_')

        # Buy/sell/hold for each coin and amount
        self.action_space = self.action_space = spaces.MultiDiscrete([3, len(coins), self.max_postitions])
        
        # (features - timestamp & balance & net worth & portfolio & initial_balance & max_postitions & positions_held) * (lookback_len - 1) 
        observation_space_len = (len(self.df.columns) - 1 + 2 + 2*(len(coins)) + 3) * (lookback_len -1) 
        self.observation_space = spaces.Box(
            low=-MAX_VALUE,
            high=MAX_VALUE, 
            shape=(observation_space_len,),
            dtype=np.float32
        )

    def step(self, action):
        assert self.positions_held[-1] >= 0 and self.positions_held[-1] <= self.max_postitions

        # Execute one time step within the environment
        t0 = perf_counter()
        reward = self._take_action(action)
        reward += self._get_base_reward()

        self.real_net_worth.append(self._calculate_real_net_worth(self.current_step))
        self.hodl_net_worth.append(self._calculate_hodl_net_worth(self.current_step))
        if self.real_net_worth[-1] > self.max_net_worth:
            self.max_net_worth = self.real_net_worth[-1]


        self.current_step += 1
        obs = self._next_observation()

        lost_90_percent_net_worth = self.real_net_worth[-1] < (self.initial_balance / 10)
        done = lost_90_percent_net_worth or self.current_step > self.max_steps
        if done:
            reward += self.real_net_worth[-1] - self.hodl_net_worth[-1]

        self.last_reward = reward
        self.cumulative_reward += reward

        info = {
            'current_step': self.current_step, 
            'last_trade': self._get_last_trade(),
            'profit': self._get_profit(),
            'fees_payed': self.fees_payed,
            'reward': reward
        }
        t1 = perf_counter()
        #print('step dt', t1-t0)
        return obs, reward, done, info


    def reset(self):
        self.current_step = self.lookback_len
        self.max_net_worth = self.initial_balance
        self.trades = []

        self.last_reward = 0
        self.cumulative_reward = 0
        self.positions_held_now = 0

        partitions, balance = self._get_initial_net_worth_distribution()

        for i, coin in enumerate(self.coins):
            self.portfolio[coin] = deque(maxlen=self.lookback_len)
            self.portfolio_value[coin] = deque(maxlen=self.lookback_len)
            self.positions_held = deque(maxlen=self.lookback_len)

            for j in range(self.lookback_len):
                coin_price = self._get_coin_avg_price(coin, j)
                coin_value = partitions[i]
                coin_amount = coin_value/coin_price
                self.portfolio[coin].append(coin_amount)
                self.portfolio_value[coin].append(coin_value)
                self.positions_held.append(self.positions_held_now)

        for i in range(self.lookback_len):
            self.balance.append(balance)
            self.real_net_worth.append(self._calculate_real_net_worth(i))
            self.hodl_net_worth.append(self._calculate_hodl_net_worth(i))

        return self._next_observation()


    def _get_initial_net_worth_distribution(self):
        partitions = []
        balance = self.initial_balance

        for _, coin in enumerate(self.coins):
            coin_value = random.uniform(0, balance)
            self.start_portfolio[coin] = coin_value/self._get_coin_avg_price(coin, self.current_step)
            if coin_value > 0:
                self.positions_held_now += 1
                balance -= coin_value
            partitions.append(coin_value)


        self.start_portfolio['balance'] = balance
        
        random.shuffle(partitions)        
        return partitions, balance


    def _get_base_reward(self):
        real_ratio = nan_to_num(sortino_ratio(diff(self.real_net_worth)), posinf=0, neginf=0)
        hodl_ratio = nan_to_num(sortino_ratio(diff(self.hodl_net_worth)), posinf=0, neginf=0)
        return (real_ratio - hodl_ratio) * 10


    def _take_action(self, action):
        action_type = action[0]
        coin = self.coins[action[1]]
        amount = action[2]/self.max_postitions

        # Set the current price to a random price within the time step
        current_price = random.uniform(self.df.at[self.current_step, coin + '_low'], self.df.at[self.current_step, coin + '_high'])

        reward = 0

        if current_price > 0: # Price is 0 before ICO
            if action_type  == 0:
                if self.positions_held[-1] == self.max_postitions:
                    self.idiot_moves += 1
                    return -1000

                # Buy amount % of balance in coin
                total_possible = self.balance[-1] / current_price
                coins_bought = max(0, total_possible * amount * (1 - self.fee))
                cost = total_possible * current_price * amount
                fee = cost * self.fee

                reward -= fee

                self.balance.append(self.balance[-1] - cost)
                self.portfolio[coin].append(self.portfolio[coin][-1] + coins_bought)
                self.fees_payed += fee

                if coins_bought > 0:
                    self.trades.append({
                        'step': self.current_step,
                        'coin': coin,
                        'coins_bought': coins_bought, 
                        'total': cost,
                        'type': 'buy',
                        'price': current_price
                    })

                self.positions_held_now += 1

            elif action_type == 1:
                if self.positions_held[-1] == 0 or self.portfolio[coin][-1] <= 0:
                    self.idiot_moves += 1
                    return -1000

                # Sell amount % of coin held
                coins_sold = max(0, self.portfolio[coin][-1] * amount)
                sell_value = coins_sold * current_price * (1 - self.fee)
                fee = coins_sold * current_price * self.fee

                reward -= fee

                self.balance.append(self.balance[-1] + sell_value)
                self.portfolio[coin].append(self.portfolio[coin][-1] - coins_sold)
                self.fees_payed += fee

                if coins_sold > 0:
                    self.trades.append({
                        'step': self.current_step,
                        'coin': coin,
                        'coins_sold': coins_sold, 
                        'total': sell_value,
                        'type': 'sell',
                        'price': current_price
                    })

                self.positions_held_now -= 1

            else:
                # Hold
                # TODO: hold within the expecation value of the variance
                reward += 10

        self.positions_held.append(self.positions_held_now)

        return nan_to_num(reward)

    def _next_observation(self):
        frame = []

        for feature in self.df.columns:
            if feature == 'timestamp': continue
            frame.append(diff(log(self.df.loc[self.current_step - self.lookback_len: self.current_step - 1, feature].values + 1)))

        # Portfolio
        for coin in self.coins:
            frame.append(diff(log(array(self.portfolio[coin]) + 1)))
            frame.append(diff(log(array(self.portfolio_value[coin]) + 1)))

        # Net worth and balance
        frame.append(diff(log(array(self.real_net_worth) + 1)))
        frame.append(diff(log(array(self.balance) + 1)))
        frame.append(array([self.initial_balance]*(self.lookback_len - 1)))
        frame.append(array([self.max_postitions]*(self.lookback_len - 1)))
        frame.append(array(self.positions_held)[1:])

        return nan_to_num(concatenate(frame), posinf=MAX_VALUE, neginf=-MAX_VALUE)


    def _calculate_real_net_worth(self, current_step):
        portfolio_value = 0
        for coin in self.coins:
            coin_price = self._get_coin_avg_price(coin, current_step)
            portfolio_value += self.portfolio[coin][-1] * coin_price
            self.portfolio_value[coin].append(portfolio_value)
        return self.balance[-1] + portfolio_value


    def _calculate_hodl_net_worth(self, current_step):
        portfolio_value = 0
        for coin in self.coins:
            coin_price = self._get_coin_avg_price(coin, current_step)
            portfolio_value += self.start_portfolio[coin] * coin_price
        return self.start_portfolio['balance'] + portfolio_value


    def _get_coin_avg_price(self, coin, step):
        return (self.df.at[step, coin + '_low'] + self.df.at[step, coin + '_high'])/2


    def _get_last_trade(self):
        try:
            return self.trades[-1]
        except IndexError:
            return None


    def _get_profit(self):
        return self.real_net_worth[-1] - self.initial_balance


    def render(self, mode='console', title=None, **kwargs):
        # Render the environment to the screen
        profit = self._get_profit()
        hodl_profit = self.hodl_net_worth[-1] - self.initial_balance      
        
        if mode == 'console':
            for coin in self.coins:
                print(coin, self.portfolio[coin][-1])

            print(f'Last_trade: {self._get_last_trade()}')
            print(f'Step: {self.current_step} of {self.max_steps}')
            print(f'Balance: {self.balance[-1]} (Initial balance: {self.initial_balance})')
            print(f'Net worth: {self.real_net_worth[-1]} (Max net worth: {self.max_net_worth})')
            print(f'Profit: {profit}')
            print('Current policy vs. hodl policy profit diff:', profit - hodl_profit)
            print(f'Fees payed: {self.fees_payed}')
            print(f'Base reward: {self._get_base_reward()}')
            print(f'Last reward: {self.last_reward}')
            print(f'Cumulative reward: {self.cumulative_reward}')
            print(f'Idiot moves: {self.idiot_moves}')
            print('--------------------------------------------------------------------------------')

        elif mode == 'human':
            if self.visualization == None:
                self.visualization = TradingGraph(self.df, self.coins, title)
            
            if self.current_step > LOOKBACK_WINDOW_SIZE:        
                self.visualization.render(self.current_step, self.real_net_worth[-1], self.trades, window_size=LOOKBACK_WINDOW_SIZE)

            print(f'Step: {self.current_step} of {self.max_steps}')
            print(f'Balance: {self.balance[-1]} (Initial balance: {self.initial_balance})')
            print(f'Net worth: {self.real_net_worth[-1]} (Max net worth: {self.max_net_worth})')
            print(f'Profit: {profit}')
            print()


    def close(self):
        if self.visualization is not None:
            self.visualization.close()
            self.visualization = None