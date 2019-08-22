import gym 
from gym.envs.registration import register
import random
import numpy as np 
import tensorflow as tf  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self, state_sp, action_sp):
        self.state_sp = state_sp
        self.action_sp = action_sp
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.target_model = self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_sp, activation='relu'))
        model.add(Dense(24, input_dim=self.state_sp, activation='relu'))
        # 각 action 에 대한 reward 값을 얻어야 하므로 linear activation 사용
        model.add(Dense(self.action_sp, activation='linear')) 
      
        model.compile(loss='mse', optimizer=tf.optimizers.Adam(lr=self.learning_rate))
        
        return model

    def predict(self, state):
        x = np.reshape(state, [1, self.state_sp])
        return self.model.predict(x)

    def predict_target(self, state):
        x = np.reshape(state, [1, self.state_sp])
        return self.target_model.predict(x)
    
    def replay(self, replay_buffer, batch_size):
        x_stack = np.empty(0).reshape(0, self.state_sp)
        y_stack = np.empty(0).reshape(0, self.action_sp)
        
        if len(replay_buffer) < batch_size:
            return
        # random sampling from experience memory
        minibatch = random.sample(replay_buffer, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            Q = self.predict(state)
            
            if done:
                Q[0][action] = reward
            else:
                Q[0][action] = reward + self.gamma * np.amax(self.predict_target(next_state))
                
            x_stack = np.vstack([x_stack, state])
            y_stack = np.vstack([y_stack, Q])
            
        history = self.model.fit(x_stack, y_stack, epochs=1, verbose=0)
        return history

def main():
    register(
            id='CartPole-v2',
            entry_point='gym.envs.classic_control:CartPoleEnv',
            tags={'wrapper_config.TimeLimie.max_eposode_steps':10000},
            reward_threshold=475.0
            )
    env = gym.make('CartPole-v2')
    
    REPLAY_MEMORY = 50000
    MAX_EPISODES = 2000
    ENOUGH_SUCCESS = 10000
    BATCH_SIZE = 16
        
    replay_buffer = deque()
    
    state_sp = len(env.observation_space.sample())  # 4
    action_sp = env.action_space.n                   # 2
    
    agent = DQNAgent(state_sp, action_sp)
    # initialize copy q-net --> target-net
    agent.target_model.set_weights(agent.model.get_weights())

    total_rewards = np.empty(MAX_EPISODES)
    best_step_cnt = 0
    
    for episode in range(MAX_EPISODES):
        eps = 1. / ((episode / 10) + 1)
        done = False
        step_cnt = 0
        state = env.reset()
    
        while not done:
            if np.random.rand(1) < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(agent.predict(state))
            
            next_state, reward, done, _ = env.step(action)
            
            if done:  # penalty
                reward -= 100
            
            replay_buffer.append(((state, action, reward, next_state, done)))
            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()
                
            state = next_state
            step_cnt += 1
            if step_cnt > ENOUGH_SUCCESS:
                break
            
        print("episode: {}/{}, step: {}".format(episode, MAX_EPISODES , step_cnt))
        
        total_rewards[episode] = step_cnt
        
        if episode % 10 == 0:
            for _ in range(50):
                history = agent.replay(replay_buffer, BATCH_SIZE)  # perform gradient descent
            print("loss = ", history.history['loss'])

            agent.target_model.set_weights(agent.model.get_weights())
        
        if step_cnt > best_step_cnt:
            best_step_cnt = step_cnt

        if step_cnt >= ENOUGH_SUCCESS:
            agent.model.save('./best_model.h5')
            break 
            
    print("best step count = ", best_step_cnt)

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.xlabel("episodes")
    plt.ylabel("steps")
    plt.show()      
    
    s = env.reset()
    
    agent = DQNAgent(state_sp, action_sp)
    
    agent.model = tf.keras.models.load_model('./best_model.h5')

    total_step =  0
    
    while True:  # best model test
        env.render()
        a = np.argmax(agent.predict(s))
        s, reward, done, _ = env.step(a)
        total_step += 1
        if done:
            print("total step = {}".format(total_step))
            break

if __name__ == '__main__':
    main()
    
   