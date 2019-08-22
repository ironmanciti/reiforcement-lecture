from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np
import random 
from collections import deque

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.__inventory = []
        self.__total_profit = 0
        self.action_history = []

        self.state_size = state_size
        self.action_size = 3   # hold, buy, sell
        self.memory = deque(maxlen=1000)
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995

        self.model = load_model("models/" + model_name) if is_eval else self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(units=32, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(units=self.action_size, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model 

    def reset(self):
        self.__inventory = []
        self.__total_profit = 0
        self.action_history = []

    def act(self, state, price_data):
        if not self.is_eval and np.random.rand() <= self.eps:
            action = random.randrange(self.action_size)
        else:
            options = self.model.predict(state)
            action = np.argmax(options[0])
        
        bought_price = None
        if action == 0:  # hold
            print('.', end='')  # flush=True
        elif action == 1:  # buy
            self.buy(price_data)
        elif action == 2: # sell
            if self.has_inventory():  # sell 할 재고 있음
                bought_price = self.sell(price_data)
            else:
                action = 0    # sell 할 재고가 없어 action 을 0 으로 변경
        else:  
            print("Invalid action !!!")

        self.action_history.append(action)

        return action, bought_price 

    def buy(self, price_data):
        self.__inventory.append(price_data)
        print("Buy: {}".format(self.format_price(price_data)))

    def sell(self, price_data):
        bought_price = self.__inventory.pop(0)
        profit = price_data - bought_price
        self.__total_profit += profit
        print("Sell: {} | profit: {}".format(self.format_price(price_data),\
                                      self.format_price(profit)))
        return bought_price

    def has_inventory(self):
        return len(self.__inventory) > 0

    def format_price(self, n):
        return ("-$" if n < 0 else "$") + "{:.2f}".format(abs(n))

    def get_total_profit(self):
        return self.format_price(self.__total_profit)

    def exprience_replay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l-batch_size+1, l):
            mini_batch.append(self.memory[i])
        
        for state, action, reward, next_state, done in mini_batch:
            if done:
                target = reward
            else:
                next_q_values = self.model.predict(next_state)[0]
                target = reward * self.gamma * np.amax(next_q_values)
            predicted_target = self.model.predict(state)
            predicted_target[0][action] = target

            self.model.fit(state, predicted_target, epochs=1, verbose=0)

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay


