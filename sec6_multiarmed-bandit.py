import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

LEARNING_RATE = 0.001

class MultiArmedBandit:

    def __init__(self):
        self.bandit = [0.2, 0.0, 0.1, -4.0]
        self.num_actions = 4

    def pull(self, arm):
        return 1 if np.random.randn(1) > self.bandit[arm] else -1


class Agent:

    def __init__(self, actions=4):
        self.num_actions = actions
        self.W = tf.Variable(tf.random.normal([self.num_actions]))
        self.model = self.model_build()
                                            
    def predict(self):
        return tf.argmax(self.W, axis=0)

    def random_or_predict(self, epsilon):
        if np.random.rand(1) < epsilon:
            return np.random.randint(self.num_actions)
        else:
            return self.predict()
        
    def model_build(self):
        model = tf.keras.models.Sequential()
        model.add(Dense(10, input_shape=(1,)))
        model.add(Dense(self.num_actions, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

env = MultiArmedBandit()
agent = Agent()
num_episodes = 50000
EPSILON = 0.3

for _ in range(num_episodes):
    action = agent.random_or_predict(EPSILON)
    reward = env.pull(action)
    a_ = tf.reduce_sum((agent.W * tf.one_hot(action, agent.num_actions)))
    agent.model.fit([a_], action)

# results time
print(env.bandit)
print(agent.W)