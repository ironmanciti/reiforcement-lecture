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

def random_search():
    env = gym.make('CartPole-v0')
    best_params = None
    best_reward = 0
    agent = Agent()

    for step in range(1000):
        agent.parameters = np.random.rand(4) * 2 - 1  # random search
        reward = run_eposode(env, agent)
        if reward > best_reward:
            best_reward = reward
            best_params = agent.parameters
            if reward >= 200:
                print('200 step achieved on step {} '.format(step))
                break
        if step % 100 == 0:
            print(reward)
    print(best_params)    # optimal policy 출력

random_search()


