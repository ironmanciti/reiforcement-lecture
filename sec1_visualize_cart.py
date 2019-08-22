import gym
import time

env = gym.make('CartPole-v0')
env.reset()         # reset() returns an initial observation
print(env.reset())  #[ 0.01916565  0.02694959  0.02015737 -0.04149654]
print(env.action_space)  # Discrete(2) left 0, right 1
# env.step() returns four values (observation, reward, done, info)
# (array([ 0.00902638, -0.24355141,  0.04793951,  0.32369867]), 1.0, False, {})  
print(env.step(0))       

for step in range(1000):
    env.render()
    env.step(env.action_space.sample())
    time.sleep(0.1)