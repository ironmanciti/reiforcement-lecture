"""
Defaultdict 를 사용하여 q_table 구현
"""
import numpy as np
from collections import defaultdict

class QLearningTable:
    def __init__(self, actions, learning_rate=0.5, reward_decay=1.0, e_greedy=0.1):
        self.actions = actions
        self.num_actions = len(self.actions)
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.eps = e_greedy
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

    def choose_action(self, observation):  # epsilon-greedy
        if np.random.uniform() < self.eps:
            return np.random.choice(self.actions)
        else:
            if np.sum(self.q_table[observation]) != 0:
                A_star = np.argmax(self.q_table[observation])
            else:
                A_star = np.random.choice(self.actions)

            policy_for_state = np.ones(self.num_actions, dtype=float) * self.eps / self.num_actions
            policy_for_state[A_star] = 1 - self.eps + self.eps / self.num_actions

            return np.random.choice(self.actions, p=policy_for_state)
        
    def learn(self, s, a, r, s_):
        q_predict = self.q_table[s][a]

        if s_ != 'terminal':
            q_target = r + self.gamma * max(self.q_table[s_])
        else:
            q_target = r 
            print('termo', self.q_table)

        self.q_table[s][a] += self.alpha * (q_target - q_predict)
        return
