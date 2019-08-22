import gym
import tensorflow as tf 
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt 
import numpy as np

env = gym.make("CartPole-v0")

input_size = env.observation_space.shape[0]   # 4
output_size = env.action_space.n              # 2
learning_rate = 0.1

model = tf.keras.models.Sequential()
model.add(Dense(output_size, input_dim=input_size, activation="linear"))
model.compile(loss="mse", optimizer="adam")

GAMMA = 0.99
eps = 0.1
num_episodes = 2000
reward_history = []

for episode in range(num_episodes):
    s = env.reset()
    step_cnt = 0
    s = np.reshape(s, [1,input_size])
    done = False

    while not done:
        step_cnt += 1
        Qs = model.predict(s)
        if np.random.rand(1) < eps:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        s_ , reward, done, _ = env.step(a)
        s_ = s = np.reshape(s_, [1,input_size])
        if done:
            Qs[0, a] = -100   # pole 이 쓰러지면 penalty 부여
        else:
            Qs_ = model.predict(s_)
            Qs[0, a] = reward + GAMMA * np.amax(Qs_)

        model.fit(s, Qs, epochs=1, verbose=0)
        s = s_ 

    reward_history.append(step_cnt)  

    print("spisode: {}, step_cnt : {}".format(episode, step_cnt))

print("Success percent :", str(sum(reward_history) / num_episodes))
plt.plot(range(len(reward_history)), reward_history)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()