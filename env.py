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
    version = 1.1

    def __init__(self, 
                df, 
                coins,
                initial_balance,
                lookback_len=1440,
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
        self.net_worth = deque(maxlen=lookback_len)
        self.portfolio = {}
        self.portfolio_value = {}

        self.trades = []
        self.postitions_avg_price = {}
        self.fees_payed = 0

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

        # Buy/sell/hold for each coin
        self.action_space = spaces.Box(low=array([-1, -1, -1], dtype=np.float32), high=array([1, 1, 1], dtype=np.float32), dtype=np.float32)
        
        # (features - timestamp & balance & net worth & portfolio) * (lookback_len - 1)
        observation_space_len = (len(self.df.columns) - 1 + 2 + 2*(len(coins))) * (lookback_len -1)
        self.observation_space = spaces.Box(
            low=-MAX_VALUE,
            high=MAX_VALUE, 
            shape=(observation_space_len,),
            dtype=np.float32
        )

    def step(self, action):
        # Execute one time step within the environment
        t0 = perf_counter()
        reward = self._take_action(action)

        self.net_worth.append(self._calculate_net_worth())
        if self.net_worth[-1] > self.max_net_worth:
            self.max_net_worth = self.net_worth[-1]

        reward += self._get_base_reward()

        self.current_step += 1
        obs = self._next_observation()

        lost_90_percent_net_worth = self.net_worth[-1] < (self.initial_balance / 10)
        done = lost_90_percent_net_worth or self.current_step > self.max_steps
        if done:
            reward += self._get_profit()

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

        partitions, balance = self._get_initial_net_worth_distribution()

        for i, coin in enumerate(self.coins):
            self.portfolio[coin] = deque(maxlen=self.lookback_len)
            self.portfolio_value[coin] = deque(maxlen=self.lookback_len)
            self.postitions_avg_price[coin] = self._get_coin_avg_price(coin, self.current_step)

            for j in range(self.lookback_len):
                coin_price = self._get_coin_avg_price(coin, j)
                coin_value = partitions[i]
                coin_amount = coin_value/coin_price
                self.portfolio[coin].append(coin_amount)
                self.portfolio_value[coin].append(coin_value)

        for i in range(self.lookback_len):
            self.balance.append(balance)
            self.net_worth.append(self.initial_balance)

        return self._next_observation()


    def _get_initial_net_worth_distribution(self):
        partitions = []
        balance = self.initial_balance

        for _ in range(len(self.coins)):
            coin_value = random.uniform(0, balance)
            balance -= coin_value
            partitions.append(coin_value)

        random.shuffle(partitions)
        return partitions, balance


    def _get_base_reward(self):
        returns = diff(self.net_worth)
        return nan_to_num(sortino_ratio(returns), posinf=0, neginf=0)


    def _take_action(self, action):
        action = nan_to_num(action, posinf=1, neginf=-1)
        action_type = action[0]
        amount = 0.5*(action[1] - 1) + 1 # Rescaling:  R_v_i = (N_max − N_min)/(O_max − O_min)∗(O_i − O_max) + N_max

        coin_action = ((len(self.coins) - 1) / 2) * (action[2] - 1) + len(self.coins) - 1
        coin = self.coins[int(coin_action)]

        # Set the current price to a random price within the time step
        current_price = random.uniform(self.df.at[self.current_step, coin + '_low'], self.df.at[self.current_step, coin + '_high'])

        reward = 0

        if current_price > 0: # Price is 0 before ICO
            if action_type  <= 1 and action_type > 1/3:
                # Buy amount % of balance in coin
                total_possible = self.balance[-1] / current_price
                coins_bought = max(0, total_possible * amount * (1 - self.fee))
                cost = total_possible * current_price * amount
                fee = cost * self.fee

                reward -= fee

                # Take position
                self.postitions_avg_price[coin] = nan_to_num((self.postitions_avg_price[coin] * self.portfolio[coin][-1] + current_price * coins_bought) \
                                                / (self.portfolio[coin][-1] + coins_bought), posinf=0, neginf=0)

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


            elif action_type >= -1 and action_type < -1/3:
                # Sell amount % of coin held
                coins_sold = max(0, self.portfolio[coin][-1] * amount)
                sell_value = coins_sold * current_price * (1 - self.fee)
                fee = coins_sold * current_price * self.fee

                reward -= fee

                # Liquidate positions
                reward += coins_sold * current_price - coins_sold * self.postitions_avg_price[coin]
                self.postitions_avg_price[coin] = nan_to_num((self.postitions_avg_price[coin] * self.portfolio[coin][-1] - current_price * coins_sold) \
                                                / (self.portfolio[coin][-1] - coins_sold), posinf=0, neginf=0)

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

            else:
                # Hold
                pass

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
        frame.append(diff(log(array(self.net_worth) + 1)))
        frame.append(diff(log(array(self.balance) + 1)))

        return nan_to_num(concatenate(frame), posinf=MAX_VALUE, neginf=-MAX_VALUE)


    def _calculate_net_worth(self):
        portfolio_value = 0
        for coin in self.coins:
            coin_price = self._get_coin_avg_price(coin, self.current_step)
            portfolio_value += self.portfolio[coin][-1] * coin_price
            self.portfolio_value[coin].append(portfolio_value)
        return self.balance[-1] + portfolio_value


    def _get_coin_avg_price(self, coin, step):
        return (self.df.at[step, coin + '_low'] + self.df.at[step, coin + '_high'])/2


    def _get_last_trade(self):
        try:
            return self.trades[-1]
        except IndexError:
            return None


    def _get_profit(self):
        return self.net_worth[-1] - self.initial_balance


    def render(self, mode='console', title=None, **kwargs):
        # Render the environment to the screen
        profit = self._get_profit()        
        
        if mode == 'console':
            for coin in self.coins:
                print(coin, self.portfolio[coin][-1])

            print(f'Last_trade: {self._get_last_trade()}')
            print(f'Step: {self.current_step} of {self.max_steps}')
            print(f'Balance: {self.balance[-1]} (Initial balance: {self.initial_balance})')
            print(f'Net worth: {self.net_worth[-1]} (Max net worth: {self.max_net_worth})')
            print(f'Profit: {profit}')
            print(f'Fees payed: {self.fees_payed}')
            print(f'Base reward: {self._get_base_reward()}')
            print(f'Last reward: {self.last_reward}')
            print(f'Cumulative reward: {self.cumulative_reward}')

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