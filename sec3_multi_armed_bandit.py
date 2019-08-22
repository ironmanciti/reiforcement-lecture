import numpy as np
import tensorflow as tf

class MultiArmedBandit:

    def __init__(self):
        self.bandit = [0.2, 0.1, 0.3, -0.1]

    def pull(self, arm):
        return 1 if np.random.uniform(0, 0.2) > self.bandit[arm] else -1

class Agent:

    def __init__(self, arms=4):
        self.parameters = np.random.rand(arms) * 2 - 1
        self.num_arms = arms 
        self.best_arm = np.argmax(self.parameters)

    def random_or_predict(self, epsilon):
        if np.random.rand(1) < epsilon:
            return np.random.randint(self.num_arms)
        else:
            return self.best_arm

env = MultiArmedBandit()
agent = Agent()
num_iter = 1000
num_arms = 4
EPSILON = 0.3
arm_reward = np.zeros(num_arms)

for i in range(num_iter):
    selected_arm = agent.random_or_predict(EPSILON)
    reward = env.pull(selected_arm)
    arm_reward[selected_arm] += reward
    agent.best_arm = np.argmax(arm_reward)

print(agent.best_arm)
print(arm_reward)
