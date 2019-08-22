import sys
sys.path.append('../')
import gym 
import lib 
import numpy as np
from sec5_policy_evaluation import policy_evaluation

GAMMA = 1.0

def policy_iteration(policy, env):
    num_states = env.nS    # 16
    num_actions = env.nA   # 4
    transitions = env.P 

    while True:
        policy_stable = True 
        V = policy_evaluation(policy, env)
        for state in range(num_states):
            old_action = np.argmax(policy[state])
            # update rule : pi_s = argmax_a(sum(p(s',r|s,a)*[r + gamma*V(s')]))
            new_action_values = np.zeros(num_actions)
            for action in range(num_actions):
                for prob, next_state, reward, _ in transitions[state][action]:
                    new_action_values[action] += \
                        prob * (reward + GAMMA * V[next_state])
            new_action = np.argmax(new_action_values)
            if new_action != old_action:
                policy_stable = False
            policy[state] = np.eye(num_actions)[new_action]
        
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V
        
if __name__ == '__main__':

    env = gym.make('GridWorld-v0')
    random_policy = np.ones([env.nS, env.nA]) * 0.25
    v_k = policy_evaluation(random_policy, env)
    print(v_k)

    optimal_policy, optimal_value = policy_iteration(random_policy, env)
    print(optimal_policy)
