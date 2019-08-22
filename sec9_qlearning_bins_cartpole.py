import gym 
import os
import sys 
import numpy as np
import pandas as pd 
from datetime import datetime 
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

class FeatureTransformer:
    def __init__(self):
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)  # 10 bins
        self.cart_velocity_bins = np.linspace(-2, 2, 9)     # 경험치
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)  # 경험치

    def transform(self, observation):
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        return build_state([
            np.digitize([cart_pos], self.cart_position_bins)[0],
            np.digitize([cart_vel], self.cart_velocity_bins)[0],
            np.digitize([pole_angle], self.pole_angle_bins)[0],
            np.digitize([pole_vel], self.pole_velocity_bins)[0]
        ])

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer

        num_states = 10 ** env.observation_space.shape[0]  # 각 state 가 10 bins 10**4
        num_actions = env.action_space.n    # 2
        self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

    def predict(self, s):
        x = self.feature_transformer.transform(s)
        return self.Q[x]   # 2 가지 action 에 대한 Q value return, shape (2,)

    def update(self, s, a, G):
        x = self.feature_transformer.transform(s)
        self.Q[x, a] += 1e-2 * (G - self.Q[x, a])  # Update using gradient descent

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            p = self.predict(s)
            return np.argmax(p)

  
def play_one_episode(model, eps, gamma):
    observation = env.reset()
    done = False 
    total_reward = 0
    iters = 0

    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, _ = env.step(action)

        total_reward += reward

        if done and iters < 200:
            reward -= 100

        #update the model
        G = reward + gamma * np.max(model.predict(observation))
        model.update(prev_observation, action, G)

        iters += 1

    return total_reward

def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

def main():
    
    ft = FeatureTransformer()
    model = Model(env, ft)
    gamma = 0.9

    N = 10000
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_one_episode(model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plot_running_avg(totalrewards)

    total_step =  0
    s = env.reset()

    while True:  # best model test
        env.render()
        a = np.argmax(model.predict(s))
        s, reward, done, _ = env.step(a)
        total_step += 1
        if done:
            print("total step = {}".format(total_step))
            return 

if __name__ == '__main__':
    main()
    

