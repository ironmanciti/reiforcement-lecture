"""
pandas DataFrame 을 이용하여 q-table 구성
"""
import numpy as np
import pandas as pd 

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.eps = e_greedy
        self.q_table = pd.DataFrame(columns=actions, dtype=np.float32)

    def choose_action(self, observation):
        self.add_state(observation)
        if np.random.uniform() < self.eps:
            return np.random.choice(self.actions)
        else:
            state_actions = self.q_table.loc[observation,:]     #['u','d','r','l'] 와 value 를 series 로 반환
            # idxmax() 는 같은 크기의 값인 경우 첫번째 index 를 반환하므로 미리 permute 하여 순서를 섞어줌
            state_actions = state_actions.reindex(np.random.permutation(state_actions.index))   
            action = state_actions.idxmax()  # 가장 value 가 큰 index 

        return action

    def learn(self, s, a, r, s_):
        self.add_state(s_)
        q_predict = self.q_table.loc[s, a]
        if s != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r 

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def add_state(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            )
