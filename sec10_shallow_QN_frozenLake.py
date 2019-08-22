import gym
import tensorflow as tf 
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt 
import numpy as np

env = gym.make("FrozenLake-v0")

input_size = env.observation_space.n 
output_size = env.action_space.n 
learning_rate = 0.1

model = tf.keras.models.Sequential()
model.add(Dense(outout_size, input_dim=input_size, activation="linear"))
model.compile(loss="mse", optimizer="adam")

GAMMA = 0.99
eps = 0.1
num_episodes = 2000
reward_history = []

for i in range(num_episodes):
    s = env.reset()
    total_reward = 0
    s = np.eye(16)[s:s+1]
    done = False

    while not done:
        
        Qs = model.predict(s)
        if np.random.rand(1) < eps:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        s_ , reward, done, _ = env.step(a)
        s_ = np.eye(16)[s_:s_ + 1] 
        if done:
            Qs[0, a] = reward
        else:
            Qs_ = model.predict(s_)
            Qs[0, a] = reward + GAMMA * np.amax(Qs_)

        model.fit(s, Qs, epochs=1, verbose=0)

        total_reward += reward
        s = s_ 

    reward_history.append(total_reward)  

    if i % 100 == 0:
        print("epoch {}/{}".format(i, num_episodes))

print("Success percent :", str(sum(reward_history) / num_episodes))
plt.plot(range(len(reward_history)), reward_history)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()