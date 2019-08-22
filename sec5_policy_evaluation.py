#Iterative Policy Evaluation
import gym
import lib
import numpy as np

GAMMA = 1.0
THETA = 1e-5

def policy_evaluation(policy, env):
    num_states = env.nS
    num_actions = env.nA
    transitions = env.P 
    # initialize an array V(s) = 0 for all s in S+
    V = np.zeros(num_states)
    while True:
        delta = 0
        for state in range(num_states):
            new_value = 0
            #update rule : V(s) = sum(pi(a|s)*sum(p(s,a)*[r + gamma*v(s')]))
            for action, prob_action in enumerate(policy[state]):
                # sum over s', r
                for prob, next_state, reward, _ in transitions[state][action]:
                    new_value += prob_action * prob * (reward + GAMMA * V[next_state])
            delta = max(delta, np.abs(new_value - V[state]))
            V[state] = new_value
        if delta < THETA:
            break
    return V 

if __name__ == '__main__':
    env = gym.make('GridWorld-v0').unwrapped
    random_policy = np.ones([env.nS, env.nA]) * 0.25
    v_k = policy_evaluation(random_policy, env)
    print(v_k.reshape(env.shape))