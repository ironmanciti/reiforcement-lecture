import tensorflow as tf 
import numpy as np
import gym

class Agent:
    def __init__(self):
        self.parameters = np.random.rand(4) * 2 - 1  # agent 가 policy 를 정하기 위한 parameter 

    def next_action(self, observations):  # agent 의 policy => random policy
        return 0 if np.matmul(self.parameters, observations) < 0 else 1   

def run_eposode(env, agent):
    observation = env.reset()
    total_reward = 0
    for _ in range(1000):
        action = agent.next_action(observation)
        observation, reward, done, info = env.step(action)  
        total_reward += reward
        if done:        
            break
    return total_reward

def hill_climbing():
    env = gym.make('CartPole-v0')
    noise_scaling = 0.1
    best_reward = 0
    agent = Agent()

    for step in range(1000):
        old_paremeters = agent.parameters
        # random search 와 다른 점 - 지금까지 best reward 를 받은 parameter 에
        # 약간의 noise 를 추가한 parameter 를 새로운 정책으로 사용
        agent.parameters = agent.parameters + (np.random.rand(4) * 2 - 1) * noise_scaling
        reward = run_eposode(env, agent)
        if reward > best_reward:
            best_reward = reward
            print('new record reward {} on step {} '.format(reward, step))
        else:
            agent.parameters = old_paremeters
        if reward >= 200:
            print('200 step achieved on step {} '.format(step))
            break

hill_climbing()



