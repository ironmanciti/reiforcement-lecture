import sys 
sys.path.append('../')
import gym 
import lib 
import numpy as np 

GAMMA = 1.0
THETA = 1e-5

def value_iteration(env):
    num_states = env.nS
    num_actions = env.nA
    transitions = env.P 

    V = np.zeros(num_states)
    while True:
        delta = 0
        for state in range(num_states):
            old_value = V[state]
            new_action_values = np.zeros(num_actions)
            for action in range(num_actions):
                for prob, next_state, reward, _ in transitions[state][action]:
                    new_action_values[action] += \
                        prob * (reward + GAMMA * V[next_state])
            V[state] = np.max(new_action_values)
            delta = np.maximum(delta, np.abs(V[state] - old_value))
        if delta < THETA:
            break
    
    # output a deterministic policy (optimal policy)
    optimal_policy = np.zeros((num_states, num_actions))
    for state in range(num_states):
        action_values = np.zeros(num_actions)
        for action in range(num_actions):
            for prob, next_state, reward, _ in transitions[state][action]:
                action_values[action] += \
                    prob * (reward + GAMMA * V[next_state])
        optimal_policy[state] = np.eye(num_actions)[np.argmax(action_values)]

    return optimal_policy, V 

if __name__ == '__main__':
    env = gym.make('GridWorld-v0')
    optimal_policy, optimal_value = value_iteration(env)

    print("Optimal Policy = ", optimal_policy)
    print("Optimal Value = ", optimal_value)
