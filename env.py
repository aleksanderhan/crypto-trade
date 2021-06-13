import gym
from gym import spaces
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import requests
import json
import random
import warnings
from collections import deque
from time import perf_counter
from queue import PriorityQueue
from empyrical import sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio

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
                lookback_len=1440,
                fee=0.005):
        
        super(CryptoTradingEnv, self).__init__()

        self.visualization = None
        self.df = df.fillna(method='bfill')
        self.coins = coins
        self.fee = fee
        self.max_steps = len(df.index) - 1
        self.lookback_len = lookback_len
        self.current_step = lookback_len
        
        self.max_initial_balance = max_initial_balance
        self.initial_balance = random.randint(1000, max_initial_balance)
        self.max_net_worth = self.initial_balance
        
        self.balance = deque(maxlen=lookback_len)
        self.net_worth = deque(maxlen=lookback_len)
        self.rewards = deque(maxlen=lookback_len)
        self.portfolio = {}
        self.portfolio_value = {}

        # Risk adjusted performance measures (whole portfolio)
        self.sharpe = deque(maxlen=lookback_len)
        self.sortino = deque(maxlen=lookback_len)
        self.calmar = deque(maxlen=lookback_len)
        self.omega = deque(maxlen=lookback_len)

        self.trades = []
        self.positions = {}
        self.fees_payed = 0

        # Buy/sell/hold for each coin
        self.action_space = spaces.Box(low=np.array([-1, -1, -1], dtype=np.float32), high=np.array([1, 1, 1], dtype=np.float32), dtype=np.float32)
        
        # (num_coins * (portfolio amount & portfolio value & candles) + (balance & net worth & timestamp & reward & risk ratios))
        observation_space_len = (len(coins) * 7 * (lookback_len -1)) + 8 * (lookback_len -1)
        self.observation_space = spaces.Box(
            low=-MAX_VALUE,
            high=MAX_VALUE, 
            shape=(observation_space_len,),
            dtype=np.float32
        )


    def step(self, action):
        t0 = perf_counter()
        # Execute one time step within the environment
        reward = self._take_action(action)
        self.rewards.append(reward)

        returns = np.diff(self.net_worth)
        self.sharpe.append(np.nan_to_num(sharpe_ratio(returns)))
        self.sortino.append(np.nan_to_num(sortino_ratio(returns)))
        self.calmar.append(np.nan_to_num(calmar_ratio(returns)))
        self.omega.append(np.nan_to_num(omega_ratio(returns)))

        self.net_worth.append(self._calculate_net_worth())
        if self.net_worth[-1] > self.max_net_worth:
            self.max_net_worth = self.net_worth[-1]

        self.current_step += 1
        obs = self._next_observation()

        lost_90_percent_net_worth = self.net_worth[-1] < (self.initial_balance / 10)
        done = lost_90_percent_net_worth or self.current_step > self.max_steps

        info = {
            'current_step': self.current_step, 
            'last_trade': self._get_last_trade(),
            'profit': self._get_profit(),
            'max_steps': self.max_steps,
            'fees_payed': self.fees_payed,
            'reward': reward
        }
        t1 = perf_counter()
        #print('step dt', t1-t0)
        return obs, reward, done, info


    def reset(self):
        # Set the current step to a random point within the data frame
        self.current_step = self.lookback_len
        self.initial_balance = random.randint(1000, self.max_initial_balance)
        self.max_net_worth = self.initial_balance
        self.trades = []

        for coin in self.coins:
            self.positions[coin] = PriorityQueue()
            self.portfolio[coin] = deque(maxlen=self.lookback_len)
            self.portfolio_value[coin] = deque(maxlen=self.lookback_len)
            for i in range(self.lookback_len):
                self.portfolio[coin].append(0)
                self.portfolio_value[coin].append(0)

        for i in range(self.lookback_len):
            self.balance.append(self.initial_balance)
            self.net_worth.append(self.initial_balance)
            self.rewards.append(0)
            self.sharpe.append(0)
            self.sortino.append(0)
            self.calmar.append(0)
            self.omega.append(0)

        return self._next_observation()        


    def _take_action(self, action):
        action = np.nan_to_num(action, posinf=1, neginf=-1)
        action_type = action[0]
        amount = 0.5*(action[1] - 1) + 1 # Rescaling:  R_v_i = (N_max − N_min)/(O_max − O_min)∗(O_i − O_max) + N_max

        coin_action = ((len(self.coins) - 1) / 2) * (action[2] - 1) + len(self.coins) - 1
        coin = self.coins[int(coin_action)]

        # Set the current price to a random price within the time step
        current_price = random.uniform(self.df.loc[self.current_step, coin + '_low'], self.df.loc[self.current_step, coin + '_high'])

        reward = 0

        if current_price > 0: # Price is 0 before ICO
            if action_type  <= 1 and action_type > 1/3:
                # Buy amount % of balance in coin
                total_possible = self.balance[-1] / current_price
                coins_bought = total_possible * amount * (1 - self.fee)
                cost = total_possible * current_price * amount
        
                self.balance.append(self.balance[-1] - cost)
                self.portfolio[coin].append(self.portfolio[coin][-1] + coins_bought)
                self.positions[coin].put((current_price, coins_bought))
                self.fees_payed += cost * self.fee

                if coins_bought > 0:
                    self.trades.append({
                        'step': self.current_step,
                        'coin': coin,
                        'coins_bought': coins_bought, 
                        'total': cost,
                        'type': 'buy',
                        'price': current_price
                    })

                    reward -= cost * self.fee

            elif action_type >= -1 and action_type < -1/3:
                # Sell amount % of coin held
                coins_sold = self.portfolio[coin][-1] * amount
                sell_value = coins_sold * current_price * (1 - self.fee)

                self.balance.append(self.balance[-1] + sell_value)
                self.portfolio[coin].append(self.portfolio[coin][-1] - coins_sold)
                self.fees_payed += coins_sold * current_price * self.fee

                if coins_sold > 0:
                    self.trades.append({
                        'step': self.current_step,
                        'coin': coin,
                        'coins_sold': coins_sold, 
                        'total': sell_value,
                        'type': 'sell',
                        'price': current_price
                    })

                    # Liquidate position(s)
                    liquidate = coins_sold
                    while liquidate > 0:
                        if not self.positions[coin].empty():
                            pos_price, pos_amount = self.positions[coin].get()
                            if liquidate > pos_amount:
                                reward += pos_amount * current_price * (1 - self.fee) - pos_amount * pos_price
                                liquidate -= pos_amount
                            elif liquidate < pos_amount:
                                reward += liquidate * current_price * (1 - self.fee) - pos_amount * pos_price
                                self.positions[coin].put((pos_price, pos_amount - liquidate))
                                liquidate = 0
                            else:
                                reward += liquidate * current_price * (1 - self.fee) - pos_amount * pos_price
                                liquidate = 0
                        else:
                            # Rounding error - trying to sell something it doesn't have
                            liquidate = 0

            else:
                # Hold
                pass

        return reward


    def _next_observation(self):
        t0 = perf_counter()
        frame = []
        for coin in self.coins:
            # Price data
            open_values = self.df.loc[self.current_step - self.lookback_len: self.current_step -1, coin + '_open'].values
            high_values = self.df.loc[self.current_step - self.lookback_len: self.current_step -1, coin + '_high'].values
            low_values = self.df.loc[self.current_step - self.lookback_len: self.current_step -1, coin + '_low'].values
            close_values = self.df.loc[self.current_step - self.lookback_len: self.current_step -1, coin + '_close'].values
            volume_values = self.df.loc[self.current_step - self.lookback_len: self.current_step -1, coin + '_volume'].values
            frame.append(np.diff(np.log(open_values)))
            frame.append(np.diff(np.log(high_values)))
            frame.append(np.diff(np.log(low_values)))
            frame.append(np.diff(np.log(close_values)))
            frame.append(np.diff(np.log(volume_values)))

            # Portfolio
            frame.append(np.diff(np.log(np.array(self.portfolio[coin]) + 1))) # +1 dealing with 0 log
            frame.append(np.diff(np.log(np.array(self.portfolio_value[coin]) + 1)))

        # Time
        timestamp_values = self.df.loc[self.current_step - self.lookback_len: self.current_step -1, 'timestamp'].values
        frame.append(np.diff(np.log(timestamp_values)))

        # Net worth and balance and rewards
        frame.append(np.diff(np.log(np.array(self.rewards) + 1)))
        frame.append(np.diff(np.log(np.array(self.balance) + 1)))
        frame.append(np.diff(np.log(self.net_worth)))

        # Risk adjusted performance ratios
        frame.append(np.diff(np.log(np.array(self.sharpe) + 1)))
        frame.append(np.diff(np.log(np.array(self.sortino) + 1)))
        frame.append(np.diff(np.log(np.array(self.calmar) + 1)))
        frame.append(np.diff(np.log(np.array(self.omega) + 1)))

        obs = np.nan_to_num(np.concatenate(frame), posinf=MAX_VALUE, neginf=-MAX_VALUE)
        t1 = perf_counter()
        #print('obs_dt', t1-t0)

        return obs


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