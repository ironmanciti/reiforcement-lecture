# Suntton p.101 On-Policy First-Visit MC control, for e-soft policies, 
# for optimum policy pi*

import gym 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

SCORE = 0  # state 내의 player 의 카드 합계 index
# ex) (6, 1, False) => (sum_hand(player), dealer open card, usable_ace 보유)
# episode 내의 array element 순서, [next_state, action, reward]
STATE = 0
ACTION = 1
REWARD = 2

EPSILON = 0.1  # for epsilon-greedy policy

def generate_episode(policy, env):
    episode = []
    cur_state = env.reset()
    while True:  # openAi gym 에서 termination 받을 때까지 episode 계속
        P = policy(cur_state)
        action_on_policy = np.random.choice(np.arange(len(P)), p=P)
        next_state, reward, done, _ = env.step(action_on_policy)  # state, reward, done, {}
        # state : ex) (6, 1, False) => (sum_hand(player), dealer open card, usable_ace 보유)
        episode.append([next_state, action_on_policy, reward])
        if done:
            return episode 
        cur_state = next_state

def onPolicy_first_visit_MC_control(env, num_episodes):
    num_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(num_actions))
    returns = defaultdict(list)
    
    def epsilon_soft_policy(state):
        A_star = np.argmax(Q[state])  # optimum policy
        # non-optimal policies
        policy_for_state = np.ones(num_actions, dtype=float) * EPSILON / num_actions
        # optimal policy
        policy_for_state[A_star] = 1 - EPSILON + EPSILON/num_actions
        return policy_for_state

    for _ in range(num_episodes):
        episode = generate_episode(epsilon_soft_policy, env)
        appeared_s_a = set([(step[STATE], step[ACTION]) for step in episode])
        for s, a in appeared_s_a:
            first_occurance = next(idx for idx, step in enumerate(episode)
                                 if step[STATE] == s and step[ACTION] == a)
            # first visit 이후의 모든 return 을 평균함
            G = sum([step[REWARD] for step in episode[first_occurance:]])
            returns[(s, a)].append(G)
            Q[s][a] = np.mean(returns[(s, a)])
    return Q

if __name__ == "__main__":
    env = gym.make("Blackjack-v0")
    Q = onPolicy_first_visit_MC_control(env, num_episodes=100000)

    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    
    X, Y = np.meshgrid(
            np.arange(12, 22), # player 가 가진 card 합계
            np.arange(1, 11)   # dealer 가 open 한 card
            )
    no_usable_ace = np.apply_along_axis(lambda idx: V[(idx[0], idx[1], False)], 2, np.dstack([X, Y]))
    usable_ace = np.apply_along_axis(lambda idx: V[(idx[0], idx[1], True)], 2, np.dstack([X, Y]))
    
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4), subplot_kw={'projection': '3d'})
    ax0.plot_surface(X, Y, no_usable_ace, cmap=plt.cm.YlGnBu_r)
    ax0.set_xlabel('Hole Cards')
    ax0.set_ylabel('Dealer')
    ax0.set_zlabel('MC Estimated Value')
    ax0.set_title('No Useable Ace')
    
    ax1.plot_surface(X, Y, usable_ace, cmap=plt.cm.YlGnBu_r)
    ax1.set_xlabel('Hole Cards')
    ax1.set_ylabel('Dealer')
    ax1.set_zlabel('MC Estimated Value')
    ax1.set_title('Useable Ace')
    
    plt.show()
