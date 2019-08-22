"""
option 1 : FrozenLake-v0 (is_slippery : True) 를 이용하여 non-deterministic(stochastic) 
           world 를 q-learning 으로 구현
option 2 : 새로운 environment 를 등록하여 deterministic world 생성
env.observation_space.n : 16
env.action_space.n : 4
action 선택은 greedy, epsilon greedy, random noise 방식의 3 가지 비교
deterministic, stochastic 환경별로 ALPHA 값과 action 선택 policy 에 따라 다른 결과 나오는 것 비교

SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
"""
import gym 
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

OPTION = 1  # 1: slippery True, 2: slippery False

GAMMA = 0.99
epsilon = 0.1
num_episodes = 2000   # option 1 인 경우 더 많은 episode 필요

if OPTION == 1:
    env = gym.make('FrozenLake-v0')   # default- is_slippery: True   
    ALPHA = 0.85
else:
    register(
        id="DeterministicFrozenLake-v0",
        entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4',
                'is_slippery': False}
    )
    env = gym.make('DeterministicFrozenLake-v0')
    ALPHA = 1.0

def epsipon_greedy_action(q_values, state, e):
    if np.random.rand(1) < e :
        action = env.action_space.sample()
    else : 
        action = greedy_action(Q[state, :])
    return action 

def greedy_action(q_values):
    maxq = np.amax(q_values)
    indices = np.nonzero(maxq == q_values)[0]
    return np.random.choice(indices)

reward_history = []
# all zero 로 Q table 초기화
Q = np.zeros([env.observation_space.n, env.action_space.n])  # 16 x 4

env.render()

for i in range(num_episodes):
    s = env.reset()    # initialize S
    total_reward = 0
    while True:        # Loop for each step of episode
        # Choose A from S using policy derived from Q 
        #a = greedy_action(Q[s,:])     # greedy 방식
        #a = epsipon_greedy_action(Q[s,:], s, epsilon)    # epsilon greedy 방식
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) / (i+1)) # random noise 방식

        s_, reward, done, _ = env.step(a)  # take action A, observe R, S'
        #env.render()
        Q[s, a] = Q[s, a] + ALPHA * (reward + GAMMA * np.max(Q[s_,:]) - Q[s,a])
        total_reward += reward
        s = s_
        if done:
            break
    reward_history.append(total_reward)

print("Success rate : "+str(sum(reward_history) / num_episodes))
print("Final Q-Table Values")
print(Q)

plt.bar(range(len(reward_history)), reward_history, color="blue")
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.show()