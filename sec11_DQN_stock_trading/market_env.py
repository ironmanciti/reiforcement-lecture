import numpy as np
import pandas as pd

class Market:
    def __init__(self, window_size, stock_name):
        self.data = self.__get_stock_data(stock_name)
        self.states = self.__get_all_window_prices_diff(self.data, window_size)
        self.index = -1
        self.last_data_index = len(self.data) - 1

    def __get_stock_data(self, stock_name):
        df = pd.read_csv("data/" + stock_name + ".csv")
        return list(df.iloc[:, 4].values)
    
    def __get_all_window_prices_diff(self, data, n):
        l = len(data)
        processd_data = []
        for t in range(l):
            state = self.__get_window(data, t, n+1)
            processd_data.append(state)
        return processd_data
    
    def __get_window(self, data, t, n):
        d = t - n + 1 
        block = data[d:t+1] if d >= 0 else -d * [data[0]] + data[0:t+1]
        result = []
        for i in range(n-1):
            result.append(block[i+1] - block[i])
        return np.array([result])

    def reset(self):
        self.index = -1
        return self.states[0], self.data[0] 

    def get_next_state_reward(self, action, bought_price=None):
        self.index += 1
        if self.index > self.last_data_index:
            self.index = 0
        next_state = self.states[self.index + 1 ]
        next_price_data = self.data[self.index + 1]

        price_data = self.data[self.index]
        reward = 0
        if action == 2 and bought_price is not None:  # selling
            # if price_data - bought_price > 0:
            #     reward = 1
            # else:
            #     reward = 0
            reward = max(price_data - bought_price, 0)

        done = True if self.index == self.last_data_index - 1 else False 

        return next_state, next_price_data, reward, done 